import numpy as np
import torch
import torch.nn as nn
from src.core import register

__all__ = ['EdgeDetector', ]

@register
class EdgeDetector(nn.Module):
    def __init__(self, num_features):
        super(EdgeDetector, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True)
        self.conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 9
        self.sobel_kernel = self.sobel_kernel.reshape((1, 1, 3, 3))
        self.sobel_kernel = np.repeat(self.sobel_kernel, 3, axis=1)
        self.sobel_kernel = np.repeat(self.sobel_kernel, 3, axis=0)
        self.conv_op.weight.data = torch.from_numpy(self.sobel_kernel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.ModuleDict({
            'stage1': nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),  # 1/2
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 1/4
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 1/8
                nn.Tanh()
            ),
            'stage2': nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 1/2
                nn.Tanh()
            ),
            'stage3': nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 1/2
                nn.Tanh()
            )
        })

    def forward(self, img):
        edge_detect = self.conv_op(img)
        edge_detect = self.relu(edge_detect)        # 原始边缘图
        edge1 = self.downsample['stage1'](edge_detect)  # 4x512xsz/8
        edge2 = self.downsample['stage2'](edge1)  # 4x1024xsz/16
        edge3 = self.downsample['stage3'](edge2)  # 4x2048xsz/32
        # print(edge_detect1.size())
        # print(edge_detect2.size())
        # print(edge_detect3.size())
        # print(edge_detect.size())

        return [edge1, edge2, edge3]
