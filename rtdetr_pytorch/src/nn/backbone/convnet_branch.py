"""
CNN Branch
"""

import torch
import torch.nn as nn
from src.core import register



__all__ = ['ConvNetBranch', ]


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        # growth_rate: 每一层增长的通道数
        # num_layers: RDB中堆叠的小卷积层数
        super(RDB, self).__init__()
        self.layers = nn.ModuleList()
        self.growth_rate = growth_rate
        channels = in_channels

        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1),
                    nn.BatchNorm2d(growth_rate),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            channels += growth_rate

        # local feature fusion
        self.lff = nn.Conv2d(channels, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        fused = self.lff(torch.cat(features, dim=1))
        return fused + x

@register
class ConvNetBranch(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4, return_idx=[0, 1, 2, 3]):
        super(ConvNetBranch, self).__init__()
        # 浅层特征提取
        self.sfe = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # RDB分支
        self.rdb1 = RDB(256, growth_rate, num_layers)
        self.rdb2 = RDB(256, growth_rate, num_layers)
        self.rdb3 = RDB(256, growth_rate, num_layers)
        self.rdb4 = RDB(256, growth_rate, num_layers)

        # 通道调整
        self.out1 = nn.Conv2d(256, 256, kernel_size=1)
        self.out2 = nn.Conv2d(256, 512, kernel_size=1)
        self.out3 = nn.Conv2d(256, 1024, kernel_size=1)
        self.out4 = nn.Conv2d(256, 2048, kernel_size=1)

        self.return_idx = return_idx

    def forward(self, x):
        feats = []

        x = self.sfe(x)         # [B, 256, H, W]
        x = self.rdb1(x)
        feat0 = self.out1(x)
        feats.append(feat0)

        x = self.rdb2(feat0)
        feat1 = self.out2(x)  # (B, 1024, H, W)
        feats.append(feat1)

        x = self.rdb3(feat1)
        feat2 = self.out3(x)  # (B, 2048, H, W)
        feats.append(feat2)

        x = self.rdb4(feat2)
        feat3 = self.out4(x)  # (B, 2048, H, W)
        feats.append(feat3)

        return [feats[i] for i in self.return_idx]
