#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
nohup python rtdetr_pytorch/tools/train.py \
    -c 'rtdetr_pytorch/configs/rtdetr/rtdetr_dual.yml' \
    -t 'rtdetr_pytorch/rtdetr_r50vd_6x_coco_from_paddle.pth' \
    > ./output/1106_0.log 2>&1 &


