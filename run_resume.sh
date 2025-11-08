#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
python rtdetr_pytorch/tools/train.py \
    -c 'rtdetr_pytorch/configs/rtdetr/rtdetr_dual.yml' \
    -r 'output/1027_3/checkpoint_best.pth'
#    > ../output/0410_0.log 2>&1 &