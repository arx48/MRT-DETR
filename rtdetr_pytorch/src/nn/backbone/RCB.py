"""
CNN Branch
"""

import torch
import torch.nn as nn



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

class ConvNetBranch(nn.Module):
    def __init__(self, in_channels, base_channels, growth_rate, num_RGB):
        super(ConvNetBranch, self).__init__()
        # 浅层特征提取
        self.sfe = nn.Sequential(
            nn.Conv2d(in_channels, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.head