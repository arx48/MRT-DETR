""""by lyuwenyu
"""
import warnings

import torch
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import datapoints

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image 
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG


__all__ = ['Compose', ]


RandomPhotometricDistort = register(T.RandomPhotometricDistort)
RandomZoomOut = register(T.RandomZoomOut)
# RandomIoUCrop = register(T.RandomIoUCrop)
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
ToImageTensor = register(T.ToImageTensor)
ConvertDtype = register(T.ConvertDtype)
SanitizeBoundingBox = register(T.SanitizeBoundingBox)
RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)

@register
class DualTransformWrapper(nn.Module):              # 让所有的transform变换支持双模图像的输入
    def __init__(self, transform_cfg: dict) -> None:
        super().__init__()
        name = transform_cfg.pop('type')
        self.transform = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**transform_cfg)

    def forward(self, inputs):
        img_vis, img_ir, target = inputs['image_vis'], inputs['image_ir'], inputs['target']
        # target["boxes"].spatial_size = F.get_image_size(img_vis)
        # 保存当前随机状态
        vis_state = torch.random.get_rng_state()

        # 变换可见光图像
        sample_vis = {"image": img_vis, **target}
        sample_vis = self.transform(sample_vis)

        # 恢复随机状态以确保相同的变换
        torch.random.set_rng_state(vis_state)

        # 变换红外图像
        sample_ir = {"image": img_ir, **target}
        sample_ir = self.transform(sample_ir)

        # 验证两个变换后的目标是否一致
        for key in target.keys():
            if not torch.equal(sample_vis[key], sample_ir[key]):
                warnings.warn(f"Target '{key}' differs between modalities after transform")

        return {
            'image_vis': sample_vis["image"],
            'image_ir': sample_ir["image"],
            'target': {k: sample_vis[k] for k in sample_vis if k != "image"}
        }


# 原始的Compose!!!!: T.Compose本质上就是把一堆变换按顺序串联起来执行，自定义Compose是为了变换可配置、可扩展
@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:                         # ops即变换列表
            for op in ops:                          # 遍历ops
                if isinstance(op, dict):            # ops如果是字典
                    name = op.pop('type')           # type对应的是具体变换类型
                    # 注册器机制，找到name（比如Resize）的类class Resize，然后用Resize(**op)实例化
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)         # 然后把这个变换加入transform列表
                    # op['type'] = name
                elif isinstance(op, nn.Module):     # op如果是一个pytorch模块实例
                    transforms.append(op)           # 直接传入写好的变换模块

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]

        super().__init__(transforms=transforms)


@register
class DualCompose(nn.Module):
    def __init__(self, ops) -> None:
        super().__init__()
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op['type']
                    if name in GLOBAL_CONFIG:
                        transforms.append(DualTransformWrapper(op))
                    else:
                        raise ValueError(f"Transform {name} not registered.")
                else:
                    raise ValueError(f"Unexpected transform config: {op}")
        else:
            transforms = [EmptyTransform()]

        self.transforms = nn.Sequential(*transforms)

    def forward(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)
        return inputs

@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    _transformed_types = (
        Image.Image,
        datapoints.Image,
        datapoints.Video,
        datapoints.Mask,
        datapoints.BoundingBox,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        
        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register
class ConvertBox(T.Transform):
    _transformed_types = (
        datapoints.BoundingBox,
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if self.out_fmt:
            spatial_size = inpt.spatial_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)
        
        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]

        return inpt

