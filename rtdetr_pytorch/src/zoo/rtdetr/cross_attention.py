'''
交叉注意力模块的构建
'''

from torch import nn
from src.core import register
import torch
import math
import torch.nn.functional as F

__all__ = ['CrossSpectrumAttention', ]


# 改进的位置编码生成函数
def generate_fixed_cosine_positional_encoding(d_model, height, width):
    pe = torch.zeros(d_model, height, width)
    # 归一化坐标
    pos_y = torch.linspace(0, 1, steps=height)
    pos_x = torch.linspace(0, 1, steps=width)
    pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
    # 频率项（分半处理
    div_term = torch.exp(torch.arange(0., d_model // 2, 2) * -(math.log(10000.0) / (d_model // 2)))
    # 混合 x/y 编码
    for i in range(0, d_model // 2, 2):
        freq = div_term[i // 2]
        pe[2 * i, :, :] = torch.sin(pos_x * freq) + torch.sin(pos_y * freq)
        pe[2 * i + 1, :, :] = torch.cos(pos_x * freq) + torch.cos(pos_y * freq)
    return pe


# 交叉注意力融合模块
@register
class CrossSpectrumAttention(nn.Module):
    def __init__(self, in_channels, num_stages):
        super(CrossSpectrumAttention, self).__init__()
        self.num_stages = num_stages
        self.attention_stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])

        self.proj = nn.Conv2d(in_channels, 64, kernel_size=1)

        # 残差连接
        self.residual = nn.Conv2d(in_channels, 64, kernel_size=1)

    def forward(self, x1, x2, weight_vis, weight_ir):
        pos_encoding1 = generate_fixed_cosine_positional_encoding(x1.size(1), x1.size(2), x1.size(3))
        pos_encoding2 = generate_fixed_cosine_positional_encoding(x2.size(1), x2.size(2), x2.size(3))

        # combined_pos_encoding = pos_encoding1 + pos_encoding2
        combined_pos_encoding = pos_encoding1
        combined_pos_encoding = combined_pos_encoding.to(x1.device)

        # 上采样到原始尺寸（H, W）
        _, _, H_feat, W_feat = x1.shape
        w_vis = F.interpolate(weight_vis.unsqueeze(1), size=(H_feat, W_feat), mode='nearest')  # [B,1,H,W]
        w_ir = F.interpolate(weight_ir.unsqueeze(1), size=(H_feat, W_feat), mode='nearest')  # [B,1,H,W]

        for i in range(self.num_stages):
            query_conv, key_conv, value_conv, softmax = self.attention_stages[i]

            # 对key和value应用权重
            x1 = x1 * w_vis
            x2 = x2 * w_ir

            query = query_conv(x1) + combined_pos_encoding
            key = key_conv(x2) + combined_pos_encoding
            value = value_conv(x2)
            attn_weights = softmax(torch.matmul(query, key.transpose(-1, -2)))
            fused_features = torch.matmul(attn_weights, value)

            # Update x1 and x2 for the next stage
            x1, x2 = fused_features, fused_features

        # fused_features = self.proj(fused_features) + self.residual(x1)     # 增加残差连接
        fused_features = self.proj(fused_features)
        return fused_features




