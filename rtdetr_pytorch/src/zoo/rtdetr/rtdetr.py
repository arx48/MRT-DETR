"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np
import cv2
import os

from src.core import register

from torch.nn import ModuleList


__all__ = ['RTDETR', ]


# 主模型：修改为双路检测
@register
class RTDETR(nn.Module):
    # __inject__ = ['backbone', 'backbone_ir', 'brightness_extractor', 'cross_attention', 'encoder', 'decoder', 'edge_extractor']
    __inject__ = ['backbone', 'backbone_ir', 'cross_attention', 'brightness_extractor', 'encoder', 'decoder']

    def __init__(self, backbone: nn.Module, backbone_ir: nn.Module, brightness_extractor, edge_extractor,
                 cross_attention: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.backbone_ir = backbone_ir
        self.brightness_extractor = brightness_extractor
        self.cross_attention = cross_attention
        # self.edge_extractor = edge_extractor
        self.decoder = decoder
        self.encoder = encoder
        # 图像的多种尺寸[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
        self.multi_scale = multi_scale

        self.num_feature_levels = 3          # S3/S4/S5

        # 降维预处理：将每层特征图投影到维度为256的hidden_dim
        input_proj_list = []
        input_proj_ir_list = []
        hidden_dim = 256
        num_channels = [512, 1024, 2048]              # PResNet
        # num_channels = [256, 512, 1024]                 # ConvNeXtv2
        for i in range(self.num_feature_levels):
            in_channels = num_channels[i]  # 当前特征层通道数
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),  # 降维
                nn.GroupNorm(32, hidden_dim),  # 标准化特征图
            ))
            input_proj_ir_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),  # 降维
                nn.GroupNorm(32, hidden_dim),  # 标准化特征图
            ))
        self.input_proj = nn.ModuleList(input_proj_list)
        self.input_proj_ir = nn.ModuleList(input_proj_ir_list)

        redim_corr_list = []
        for i in range(self.num_feature_levels):
            redim_corr_list.append(nn.Sequential(
                nn.Conv2d(2 * hidden_dim + 64, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)))
        self.redim_corr = nn.ModuleList(redim_corr_list)

    def forward(self, x1, x2, targets=None, save_dir=None):

        weight_vis, weight_ir, brightness = self.brightness_extractor(x1, x2)

        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x1 = F.interpolate(x1, size=[sz, sz])
            x2 = F.interpolate(x2, size=[sz, sz])

        x1, f_vis = self.backbone(x1, mode='vis')
        x2, f_ir, aux_loss = self.backbone_ir(x2, ref_feats=f_vis, mode='ir')

        pro_x1, pro_x2 = [], []

        batch_size = x1[0].shape[0]

        for i in range(self.num_feature_levels):
            pro_x1_layer = self.input_proj[i](x1[i])
            pro_x2_layer = self.input_proj_ir[i](x2[i])

            # 将亮度权重上采样到当前层，并直接作用于两路特征
            Hf, Wf = pro_x1_layer.shape[-2:]
            wv = F.interpolate(weight_vis.unsqueeze(1), size=(Hf, Wf), mode='nearest')  # [B,1,Hf,Wf]
            wi = F.interpolate(weight_ir.unsqueeze(1), size=(Hf, Wf), mode='nearest')  # [B,1,Hf,Wf]
            pro_x1_layer = pro_x1_layer * wv
            pro_x2_layer = pro_x2_layer * wi
            pro_x1.append(pro_x1_layer)
            pro_x2.append(pro_x2_layer)

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                for b in range(batch_size):
                    for j, feat in enumerate([pro_x1_layer, pro_x2_layer]):
                        img = feat[b].mean(dim=0).detach().cpu().numpy()  # 通道平均
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                        img = img.astype(np.uint8)
                        # img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                        name = f"layer{i}_{'VIS' if j == 0 else 'IR'}_batch{b}.png"
                        cv2.imwrite(os.path.join(save_dir, name), img)

        x_fused = []
        for i in range(self.num_feature_levels):
            fused_feature = self.cross_attention(pro_x1[i], pro_x2[i], weight_vis, weight_ir)
            corr = self.redim_corr[i](torch.cat((pro_x1[i], pro_x2[i], fused_feature), 1))
            x_fused.append(corr)

            if save_dir is not None:
                for b in range(batch_size):
                    img = corr[b].mean(dim=0).detach().cpu().numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                    img = img.astype(np.uint8)
                    img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    name = f"layer{i}_FUSED_batch{b}.png"
                    cv2.imwrite(os.path.join(save_dir, name), img_color)

        x = self.encoder(x_fused)
        x = self.decoder(x, targets)

        if self.training:
            return x, aux_loss
        else:
            return x
        

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
