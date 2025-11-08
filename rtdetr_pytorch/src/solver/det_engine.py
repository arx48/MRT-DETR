"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable
from src.misc import dist

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)
import time
from ptflops import get_model_complexity_info
from thop import profile
from pathlib import Path


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()           # 训练模式
    # print("model:", model)                # 打印模型架构参数
    criterion.train()       # 损失函数为训练模式
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)       # 隔10个batch打印一次训练日志
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, samples_ir, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        samples_ir = samples_ir.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]        # k是标签的名称，v是标签的数据
        # torch.cuda.synchronize()
        # t0 = time.time()

        if scaler is not None:   # 判断是否使用了混合精度训练（默认无）：scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs, aux_losses = model(samples, samples_ir, targets)           # 前向传播执行
                # outputs = model(samples, targets)
                # print("outputs:", outputs)
                # print("type(outputs):", type(outputs))
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, aux_losses)         # 精度计算
            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs, aux_losses = model(samples, samples_ir, targets)
            # outputs = model(samples, samples_ir, targets)
            loss_dict = criterion(outputs, targets, aux_losses)
            # loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)             # 各类loss的输出
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, epoch):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )
    inference_start_time = time.time()          # 记录每轮推理时间

    # dummy_rgb = torch.randn(1, 3, 640, 640).to(device)
    # dummy_ir = torch.randn(1, 3, 640, 640).to(device)
    # # 确保模型在 eval 模式、在计算设备上
    # model.eval().to(device)
    #
    # # 计算 FLOPs 和参数
    # flops, params = profile(model, inputs=(dummy_rgb, dummy_ir), verbose=False)
    #
    # print(f"[FLOPs] {flops / 1e9:.2f} GFLOPs")
    # print(f"[Params] {params / 1e6:.2f} M")

    for samples, samples_ir, targets in metric_logger.log_every(data_loader, 10, header):
    # for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        samples_ir = samples_ir.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples, samples_ir)
        # outputs = model(samples)

        # print(outputs['pred_logits'].shape)
        # output是一个字典，包括pred_logits和pred_boxes
        # pred_logits：分类的原始输出[bs, num_queries, num_classes]，每一张图片，对每一个query预测属于每个类别的原始置信度（没做softmax）
        # pred_boxes: 预测的bounding box，形状是[bs, num_queries, 4]
        # print("outputs:", outputs)
        # print("type(outputs):", type(outputs))

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)                        # 经过postprocessor后的最终输出
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

            # ====== 新增：保存预测框到txt文件 ======
            for target, output in zip(targets, results):
                image_id = target["image_id"].item()
                # 从 COCO dataset 获取 file_name
                file_name = base_ds.loadImgs(image_id)[0]["file_name_RGB"]
                stem = Path(file_name).stem

                # 创建保存路径
                save_dir = Path(output_dir) / "FLIR"
                save_dir.mkdir(parents=True, exist_ok=True)
                pred_txt_path = save_dir / f"{stem}.txt"

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                h, w = target["orig_size"].cpu().numpy()
                # boxes是 xyxy 格式，转为 xywh 中心点坐标
                with open(pred_txt_path, 'w') as f:
                    for box, score, cls in zip(boxes, scores, labels):
                        if score > 0.5:  # 阈值可调
                            x_min, y_min, x_max, y_max = box
                            x_center = (x_min + x_max) / 2
                            y_center = (y_min + y_max) / 2
                            width = x_max - x_min
                            height = y_max - y_min
                            f.write(f"{cls},{x_center:.2f},{y_center:.2f},{width:.2f},{height:.2f},{score:.6f}\n")

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # 计算推理时间和帧率
    inference_time = time.time() - inference_start_time
    inference_fps = len(data_loader.dataset) / inference_time

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images 计算预测精度
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    stats['inference_time_sec'] = inference_time
    stats['inference_fps'] = inference_fps
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    # 把指标写入.txt文件
    if dist.is_main_process():
        eval_log_path = output_dir / "evaluation_results.txt"
        with open(eval_log_path, "a") as f:
            f.write(f"Epoch: {epoch}\n")
            for k, v in stats.items():
                if isinstance(v, list):
                    f.write(f"{k}:\n")
                    for i, metric in enumerate(v):
                        f.write(f"  metric_{i}: {metric:.4f}\n")
                else:
                    f.write(f"{k}: {v}\n")
            f.write("\n")

    return stats, coco_evaluator



