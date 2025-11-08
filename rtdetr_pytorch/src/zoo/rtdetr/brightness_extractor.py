import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

from src.core import register

__all__ = ['BrightnessExtractor', ]


@register
class BrightnessExtractor(nn.Module):
    """
    同时建模全局与局部亮度的引导权重提取模块。
    输入：RGB图像
    输出：每个patch的可见光/红外权重图
    """
    def __init__(self, patch_size=16, hidden_dim=32, beta=0.6, day_night_range=(0.2, 0.8)):
        super(BrightnessExtractor, self).__init__()
        self.patch_size = patch_size
        self.beta = beta
        self.a, self.c = day_night_range
        self.t1, self.t2 = 0.45, 0.55
        self.local_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.global_fc = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.index = 0

    def forward(self, x_rgb, x_ir):
        """
        Args:
            x_rgb: [B, 3, H, W] 可见光图像
        Returns:
            w_vis: [B, H_patch, W_patch] 可见光权重
            w_ir:  [B, H_patch, W_patch] 红外权重
            brightness_map: [B, H_patch, W_patch] patch亮度图（可视化用）
        """

        # 原始
        B, C, H, W = x_rgb.shape
        p = self.patch_size

        # 转换为灰度图
        gray = 0.2989 * x_rgb[:, 0] + 0.5870 * x_rgb[:, 1] + 0.1140 * x_rgb[:, 2]  # [B, H, W]
        gray = gray.unsqueeze(1)  # [B, 1, H, W]

        # 计算全局亮度均值
        global_mean = gray.mean(dim=[2, 3], keepdim=True)  # [B, 1, 1, 1]
        global_feat = self.global_fc(global_mean.view(B, 1))  # [B, hidden_dim]
        global_feat = global_feat.view(B, -1, 1, 1)  # [B, hidden_dim, 1, 1]

        # padding 保证整除
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        gray_padded = F.pad(gray, (0, pad_w, 0, pad_h), mode='reflect')     # podding补齐尺寸
        H_pad, W_pad = gray_padded.shape[-2:]

        # 提取局部亮度特征
        local_feat = self.local_encoder(gray_padded)  # [B, hidden_dim, H', W']

        # 全局特征 broadcast 并拼接
        global_feat = global_feat.expand(-1, -1, H_pad, W_pad)  # [B, hidden_dim, H', W']
        fused_feat = torch.cat([local_feat, global_feat], dim=1)  # [B, 2*hidden_dim, H', W']

        # MLP → 权重图
        weight_map = self.mlp(fused_feat).squeeze(1)  # [B, H', W']
        weight_map = weight_map[:, :H_pad//p * p, :W_pad//p * p]  # 裁剪冗余边缘

        # 平均池化到 patch 尺寸
        w_vis = F.avg_pool2d(weight_map.unsqueeze(1), kernel_size=p, stride=p).squeeze(1)  # [B, H_patch, W_patch]
        w_ir = 1 - w_vis

        # 可视化亮度均值图
        brightness_map = F.avg_pool2d(gray_padded, kernel_size=p, stride=p).squeeze(1)  # [B, H_patch, W_patch]
        # self.visualize_brightness_weight(x_rgb[0], x_ir[0], w_vis[0], self.index, save_dir='./0809_vis/brightness')
        self.index += 1

        # 全局先验校准（分段线性）：增强昼/夜分离度
        b_scalar = brightness_map.mean(dim=(1, 2), keepdim=True)                      # [B,1]
        denom = (self.t2 - self.t1) + 1e-6
        alpha = (b_scalar - self.t1) / denom                                          # [B,1]
        alpha = torch.clamp(alpha, 0.0, 1.0)
        p_vis = self.a + (self.c - self.a) * alpha                                    # [B,1]
        p_vis_map = p_vis.expand(-1, brightness_map.size(1), brightness_map.size(2))  # [B,Hp,Wp]
        w_vis = (1 - self.beta) * w_vis + self.beta * p_vis_map
        w_ir = 1 - w_vis

        return w_vis, w_ir, brightness_map

    def visualize_brightness_weight(self, original_rgb, original_ir, weight_map, index=None, save_dir='./brightness_vis'):
        """
        在原图上可视化 w_vis（亮度感知可见光权重）。

        Args:
            original_rgb: [3, H, W] tensor, 原始可见光图像
            w_vis: [H_patch, W_patch] tensor，可见光 patch 权重图
            patch_size: int, 每个 patch 的大小
            save_path: 可选，保存路径
        """

        os.makedirs(save_dir, exist_ok=True)

        # 处理原图
        img = original_rgb.detach().cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # 归一化到[0,1]

        img_ir = original_ir.detach().cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
        img_ir = (img_ir - img_ir.min()) / (img_ir.max() - img_ir.min() + 1e-6)  # 归一化到[0,1]

        H, W = img.shape[:2]

        # 处理权重图
        weight_map = weight_map.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
        weight_resized = F.interpolate(weight_map, size=(H, W), mode='bilinear', align_corners=False)
        weight_resized = weight_resized.squeeze().cpu().numpy()  # [H,W]

        # 生成热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * weight_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0  # 转为RGB格式

        overlay = img * 0.6 + heatmap * 0.4
        overlay = np.clip(overlay, 0, 1)

        # 保存路径
        if index is not None:
            name = f"sample_{index:04d}.png"
        else:
            name = "sample.png"

        # 保存三张图（可根据需要选择保存）
        plt.imsave(os.path.join(save_dir, f"{name}_vis.jpg"), img)
        plt.imsave(os.path.join(save_dir, f"{name}_ir.jpg"), img_ir)
        plt.imsave(os.path.join(save_dir, f"{name}_weightmap.jpg"), weight_resized, cmap='jet')
        # plt.imsave(os.path.join(save_dir, f"overlay_{name}"), overlay)