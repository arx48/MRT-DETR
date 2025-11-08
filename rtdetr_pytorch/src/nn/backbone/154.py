'''by lyuwenyu
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d

from src.core import register


__all__ = ['PResNet']


ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}


donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in, 
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1, 
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    act=act)
            )

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


@register
class PResNet(nn.Module):
    def __init__(
        self, 
        depth, 
        variant='d', 
        num_stages=4, 
        return_idx=[0, 1, 2, 3], 
        act='relu',
        freeze_at=-1, 
        freeze_norm=True, 
        pretrained=False):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            self.load_state_dict(state)
            print(f'Load PResNet{depth} state_dict')

        self.offset1 = DeformableModule(in_channels=256, num_keypoints=5, offset_scale=6.0)
        # self.offset2 = DeformableModule(in_channels=512, num_keypoints=9, offset_scale=3.0)
            
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x, mode='vis', ref_feats=None):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        feats = []           # 中间特征：用来生成偏移量
        loss = {}
        # for idx, stage in enumerate(self.res_layers):
        #     x = stage(x)
        #     if idx in self.return_idx:
        #         outs.append(x)
        # return outs
        if mode == 'vis':               # 可见光特征
            for idx, stage in enumerate(self.res_layers):
                x = stage(x)
                if idx in self.return_idx:
                    outs.append(x)
                feats.append(x)
            return outs, feats

        elif mode == 'ir':             # 红外特征
            assert ref_feats is not None, "IR 模式需要传入可见光参考特征 ref_feats"
            for idx, stage in enumerate(self.res_layers):
                x = stage(x)
                if idx == 0:
                    offset_x, offset_y, attn_weights = self.offset1(ref_feats[0], x)           # 得到偏移量
                    x_offset, sampled_features, attention_weights = self.offset1.deform_with_attention(x, offset_x, offset_y, attn_weights)
                    # self.visualize_deformation(ref_feats[0], x, x_offset, offset_x, offset_y, attn_weights,
                    #                            './output/test/visualize')
                    # offset_x, offset_y = self.offset1(ref_feats[0], x)  # 得到偏移量
                    # x_offset, sampled_features = self.offset1.deform_with_offset(x, offset_x, offset_y)
                    # self.visualize_deformation(ref_feats[0], x, x_offset, offset_x, offset_y, attn_weights,
                    #                            './output/test/visualize')
                    x = x_offset
                    loss = self.offset1.loss_calculate(ref_feats[0], offset_x, offset_y, x, attention_weights)
                # elif idx == 1:
                #     offset_x, offset_y, attn_weights = self.offset2(ref_feats[1], x)
                #     # self.visualize_deformation(x0[1], x, offset_x, offset_y, attn_weights, './output/0526_2/visualize')
                #     x_offset, sampled_features, attention_weights = self.offset2.deform_with_attention(x, offset_x, offset_y, attn_weights)
                #     x = x_offset
                if idx in self.return_idx:
                    outs.append(x)
                feats.append(x)
            return outs, feats, loss
        else:
            print('mode is invalid!')

    def visualize_deformation(self, x1, x2, x2_o, offset_x, offset_y, attention_weights, save_path=None):
        """可视化变形场和注意力权重"""
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import os

        # 创建图像网格
        # num_rows = 3 if attention_np is not None else 2
        # fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 分别可视化可见光特征图、偏移前红外特征图、偏移后红外特征图
        vis_feat = x1[0].detach().mean(0).cpu().numpy()
        im0 = axes[0].imshow(vis_feat, cmap='viridis')
        axes[0].set_title('Visible Features')            # 可见光特征图
        axes[0].axis('off')
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax)

        ir_feat = x2[0].detach().mean(0).cpu().numpy()
        im1 = axes[1].imshow(ir_feat, cmap='viridis')
        axes[1].set_title('IR Features')                 # 红外特征图
        axes[1].axis('off')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)

        ir_feat2 = x2_o[0].detach().mean(0).cpu().numpy()
        im2 = axes[2].imshow(ir_feat2, cmap='viridis')
        axes[2].set_title('Warped IR Features')
        axes[2].axis('off')
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)

        # 偏移场
        _, _, H, W = x1.shape
        ox = offset_x[0]  # [K, H, W]
        oy = offset_y[0]  # [K, H, W]
        attn = attention_weights[0]  # [K, H, W]
        attn = attn / (attn.sum(dim=0, keepdim=True) + 1e-8)
        # 加权融合偏移
        fused_offset_x = (attn * ox).sum(dim=0)  # [H, W]
        fused_offset_y = (attn * oy).sum(dim=0)  # [H, W]
        # 偏移可视化图准备
        offset_x_np = fused_offset_x.detach().cpu().numpy() * (W / 2)
        offset_y_np = fused_offset_y.detach().cpu().numpy() * (H / 2)
        offset_magnitude = np.sqrt(offset_x_np ** 2 + offset_y_np ** 2)
        avg_magnitude = np.mean(offset_magnitude)
        max_magnitude = np.max(offset_magnitude)

        y, x = np.mgrid[0:H, 0:W]
        step = max(1, H // 20)
        axes[3].quiver(
            x[::step, ::step], y[::step, ::step],
            offset_x_np[::step, ::step],
            offset_y_np[::step, ::step],
            angles='xy', scale_units='xy', scale=0.5,
            color='blue', width=0.003
        )
        axes[3].imshow(ir_feat, cmap='gray', alpha=0.3)
        axes[3].invert_yaxis()
        axes[3].set_xlim(0, W)
        axes[3].set_ylim(H, 0)
        axes[3].set_title(f'Offset Field (Avg: {avg_magnitude:.4f}, Max: {max_magnitude:.4f})')

        # # 计算偏移量的振幅
        # _, _, H, W = x1.size()
        # offset_x_np = (W/2) * offset_x_np
        # offset_y_np = (H/2) * offset_y_np
        # offset_magnitude = np.sqrt(offset_x_np ** 2 + offset_y_np ** 2)
        # avg_magnitude = np.mean(offset_magnitude)
        # max_magnitude = np.max(offset_magnitude)
        #
        # # 可视化偏移量振幅热图（平均所有关键点）
        # im2 = axes[0, 2].imshow(np.mean(offset_magnitude, axis=0), cmap='hot')
        # axes[0, 2].set_title(f'Offset Magnitude (Avg: {avg_magnitude:.4f}, Max: {max_magnitude:.4f})')
        # axes[0, 2].axis('off')
        # divider = make_axes_locatable(axes[0, 2])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im2, cax=cax)
        #
        # # 可视化变形场 (从不同的关键点采样)
        # h, w = offset_x_np.shape[1:]
        # y, x = np.mgrid[0:h, 0:w]
        #
        # # 下采样以便更清晰地显示
        # step = max(1, h // 20)  # 动态调整步长，确保向量数量合适
        #
        # # 为不同关键点可视化变形场
        # for i in range(min(3, offset_x_np.shape[0])):
        #     axes[1, i].quiver(x[::step, ::step], y[::step, ::step],
        #                       offset_x_np[i, ::step, ::step], offset_y_np[i, ::step, ::step],
        #                       angles='xy', scale_units='xy', scale=0.5,
        #                       color='r', width=0.003)
        #     # 添加背景图像
        #     axes[1, i].imshow(ir_feat, cmap='gray', alpha=0.3)
        #     axes[1, i].set_title(f'Deformation Field (Keypoint {i + 1})')
        #     axes[1, i].invert_yaxis()
        #     axes[1, i].set_xlim(0, w)
        #     axes[1, i].set_ylim(h, 0)
        #
        # # 可视化注意力权重 (如果有)
        # if attention_np is not None:
        #     for i in range(min(3, attention_np.shape[0])):
        #         im = axes[2, i].imshow(attention_np[i], cmap='hot')
        #         axes[2, i].set_title(f'Attention Weight (Keypoint {i + 1})')
        #         axes[2, i].axis('off')
        #         divider = make_axes_locatable(axes[2, i])
        #         cax = divider.append_axes("right", size="5%", pad=0.05)
        #         plt.colorbar(im, cax=cax)

        plt.tight_layout()
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            import time
            timestamp = int(time.time())
            filename = f"{save_path}/deform_viz_{timestamp}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图像已保存至 {filename}")
        plt.close(fig)
        return fig


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
            # nn.Softmax(dim=1)
        )
        # self.scale = nn.Parameter(torch.tensor(1.0))

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

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
            padding_mode='border',
            align_corners=True
        )  # [B*K, C, H, W]

        # 重塑回原始维度
        sampled_features = sampled_features.reshape(B, K, C, H, W)  # [B, K, C, H, W]

        # # 应用注意力权重进行加权融合
        # attention_weights = attention_weights.unsqueeze(2)  # [B, K, 1, H, W]
        # # attention_weights = attention_weights * self.scale
        # # attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(2)
        # weighted_features = sampled_features * attention_weights
        # fused_features = weighted_features.sum(dim=1)  # [B, C, H, W]

        # 不应用注意力权重
        fused_features = sampled_features.sum(dim=1)  # [B, C, H, W]

        return fused_features, sampled_features, attention_weights

    def loss_calculate(self, x1, offset_x, offset_y, x2, attention_weights):
        aux_losses = {}
        # # 偏移量正则化损失
        # offset_loss = torch.mean(torch.abs(offset))
        # aux_losses['loss_deform_offset'] = offset_loss
        #
        # # 根据特征相似度计算对齐损失
        # align_loss = F.mse_loss(x2, x1)
        # aux_losses['loss_deform_align'] = align_loss
        #
        # # 计算注意力多样性损失
        # attn_diversity_loss = -torch.mean(torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1))
        # aux_losses['loss_deform_attn'] = attn_diversity_loss

        # 控制偏移量大小（暂时不加入损失项）
        # offset_magnitude = torch.sqrt(offset_x ** 2 + offset_y ** 2)
        # offset_loss = offset_magnitude.mean()
        # aux_losses['loss_deform_offset'] = offset_loss

        # 使用余弦相似度对齐
        cosine_sim = F.cosine_similarity(x2, x1, dim=1)  # [B, H, W]
        align_loss = 1 - cosine_sim.mean()  # 越接近1越相似
        aux_losses['loss_deform_align'] = align_loss

        # # 注意力熵损失
        # attn_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1)  # [B, H, W]
        # attn_entropy_loss = attn_entropy.mean()
        # aux_losses['loss_deform_attn_entropy'] = attn_entropy_loss

        return aux_losses

        # # PFC:点级特征一致性损失
        # B, C, H, W = x1.shape
        # N = 128     # 随机点数
        # device = x1.device
        #
        # # flatten: [B, H*W, C]
        # x1_flat = x1.permute(0, 2, 3, 1).reshape(B, -1, C)
        # x2_flat = x2.permute(0, 2, 3, 1).reshape(B, -1, C)
        #
        # idx = torch.randperm(H * W, device=device)[:N]  # 随机抽取点对两个模态采样
        # x1_sample = x1_flat[:, idx, :]  # [B, N, C]
        # x2_sample = x2_flat[:, idx, :]  # [B, N, C]
        #
        # # 初始化 scale 参数
        # logit_scale = self.logit_scale.exp()
        #
        # total_loss = 0
        # for i in range(B):
        #     anchor = F.normalize(x1_sample[i], dim=1)  # [N, C]
        #     positive = F.normalize(x2_sample[i], dim=1)  # [N, C]
        #     # 计算相似度矩阵
        #     logits = logit_scale * anchor @ positive.T  # [N, N]
        #     # 对称损失
        #     labels = torch.arange(N, device=device)
        #     loss_i = (F.cross_entropy(logits, labels) +
        #               F.cross_entropy(logits.T, labels)) / 2
        #     total_loss += loss_i
        #
        # return {'loss_pfc': total_loss / B}

        # # mobius加法
        # def mobius_add(x, y, c=1.0):
        #     x2 = (x ** 2).sum(dim=-1, keepdim=True)
        #     y2 = (y ** 2).sum(dim=-1, keepdim=True)
        #     xy = (x * y).sum(dim=-1, keepdim=True)
        #     num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        #     denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        #     return num / (denom + 1e-5)
        #
        # # 双曲空间映射函数
        # def exp_map_poincare(x, base_point=None):
        #     c = torch.tensor(1.0)
        #     norm = torch.norm(x, dim=-1, keepdim=True)
        #     norm = torch.clamp(norm, min=1e-5)
        #     scaled = torch.tanh(torch.sqrt(c) * norm) * x / (torch.sqrt(c) * norm)
        #     if base_point is None:
        #         base_point = torch.zeros_like(x)
        #     return mobius_add(base_point, scaled, c)
        #
        # # 双曲空间对比损失
        # B, C, H, W = x1.shape
        # device = x1.device
        #
        # x1_flat = x1.permute(0, 2, 3, 1).reshape(B, -1, C)
        # x2_flat = x2.permute(0, 2, 3, 1).reshape(B, -1, C)
        #
        # N = 128
        # idx = torch.randperm(H * W, device=device)[:N]
        # x1_samples = x1_flat[:, idx, :]  # [B, N, C]
        # x2_samples = x2_flat[:, idx, :]  # [B, N, C]
        #
        # # 映射到双曲空间
        # x1_hyper = exp_map_poincare(x1_samples)  # [B, N, C]
        # x2_hyper = exp_map_poincare(x2_samples)  # [B, N, C]
        #
        # # 温度参数
        # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(device)
        # logit_scale = logit_scale.exp()
        #
        # total_loss = 0
        # for i in range(B):
        #     anchor = F.normalize(x1_hyper[i], dim=1)  # [N, C]
        #     positive = F.normalize(x2_hyper[i], dim=1)  # [N, C]
        #
        #     # 双曲空间下相似度近似为点乘（因为已经tanh归一化）
        #     logits = logit_scale * anchor @ positive.T
        #     labels = torch.arange(N, device=device)
        #     loss_i = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        #     total_loss += loss_i
        #
        # return {'loss_pfc': total_loss / B}
