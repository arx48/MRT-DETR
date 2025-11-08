'''
可变形的偏移量预测模块
'''

from torch import nn
from src.core import register
import torch
import math
import torch.nn.functional as F

__all__ = ['DeformableModule', ]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


@register
class DeformableModule(nn.Module):
    def __init__(self, in_channels, num_keypoints=9, offset_scale=0.1):
        super(DeformableModule, self).__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.offset_scale = offset_scale

        self.offset_predict = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.res_block1 = ResidualBlock(in_channels, in_channels)
        self.bottle_block1 = BottleneckBlock(in_channels, in_channels)
        self.res_block2 = ResidualBlock(in_channels, in_channels)

        self.offset_head = nn.Conv2d(in_channels, 2*num_keypoints, kernel_size=3, padding=1)
        self.attention_head = nn.Sequential(
            nn.Conv2d(in_channels, num_keypoints, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.constant_(self.offset_head.weight, 0)
        if self.offset_head.bias is not None:
            nn.init.constant_(self.offset_head.bias, 0)

        # 初始化注意力权重，使其开始时大致均匀
        for m in self.attention_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1.0 / self.num_keypoints)

    def create_grid(self, h, w):
        # 创建标准网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, dtype=torch.float32),
            torch.linspace(-1, 1, w, dtype=torch.float32)
        )
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)
        return grid

    def forward(self, x1, x2):
        # 拼接两个特征图
        offset_input = torch.cat([x1, x2], dim=1)

        # 通过增强的网络处理
        feat = self.offset_predict(offset_input)
        feat = self.res_block1(feat)
        feat = self.bottle_block1(feat)
        feat = self.res_block2(feat)

        # 预测偏移量
        offset = self.offset_head(feat)     # [B, 2*K, H, W]

        # 控制像素偏移范围
        offset = torch.tanh(offset) * self.offset_scale

        # 预测注意力权重
        attention_weights = self.attention_head(feat)       # [B, K, H, W]

        # 重塑偏移量
        B, _, H, W = offset.size()
        offset = offset.view(B, self.num_keypoints, 2, H, W)
        offset_x = offset[:, :, 0, :, :]  # [B, K, H, W]
        offset_y = offset[:, :, 1, :, :]  # [B, K, H, W]
        offset_x = offset_x / ((W-1) / 2)
        offset_y = offset_y / ((H-1) / 2)               # 输出归一化后的偏移量，便于后面网格处理

        # print('attention_weights', attention_weights.size())
        # print(attention_weights)

        return offset_x, offset_y, attention_weights

    def deform_with_attention(self, feature_map, offset_x, offset_y, attention_weights):
        B, C, H, W = feature_map.size()
        K = offset_x.size(1)  # 关键点个数

        # 创建基础采样网格：归一化的坐标
        grid = self.create_grid(H, W).to(feature_map.device)  # [1, H, W, 2]

        # 扩展网格以适应多个关键点
        grid_x = grid[..., 0].unsqueeze(1).expand(B, K, H, W)  # [B, K, H, W]
        grid_y = grid[..., 1].unsqueeze(1).expand(B, K, H, W)  # [B, K, H, W]

        # 应用偏移
        grid_x = grid_x + offset_x  # [B, K, H, W]
        grid_y = grid_y + offset_y  # [B, K, H, W]

        # 合并为采样网格
        sampling_grid = torch.stack((grid_x, grid_y), dim=-1)  # [B, K, H, W, 2]

        # 确保采样点不出界
        sampling_grid = torch.clamp(sampling_grid, -1, 1)

        # 扩展特征图以适应多个关键点
        feature_map_expanded = feature_map.unsqueeze(1).expand(B, K, C, H, W)  # [B, K, C, H, W]

        # 重塑以便于使用grid_sample
        feature_map_reshaped = feature_map_expanded.reshape(B * K, C, H, W)
        sampling_grid_reshaped = sampling_grid.reshape(B * K, H, W, 2)

        # 执行采样
        sampled_features = F.grid_sample(
            feature_map_reshaped,
            sampling_grid_reshaped,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # [B*K, C, H, W]

        # 重塑回原始维度
        sampled_features = sampled_features.reshape(B, K, C, H, W)  # [B, K, C, H, W]

        # 应用注意力权重进行加权融合
        attention_weights = attention_weights.unsqueeze(2)  # [B, K, 1, H, W]
        weighted_features = sampled_features * attention_weights
        fused_features = weighted_features.sum(dim=1)  # [B, C, H, W]

        return fused_features, sampled_features, attention_weights

    def loss_calculate(self, x1, offset, fused_offset, attention_weights):
        aux_losses = {}
        # 偏移量正则化损失
        offset_loss = torch.mean(torch.abs(offset))
        aux_losses['loss_deform_offset'] = offset_loss

        # 根据特征相似度计算对齐损失
        align_loss = F.mse_loss(fused_offset, x1)
        aux_losses['loss_deform_align'] = align_loss

        # 计算注意力多样性损失
        attn_diversity_loss = -torch.mean(torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1))
        aux_losses['loss_deform_attn'] = attn_diversity_loss

        return aux_losses