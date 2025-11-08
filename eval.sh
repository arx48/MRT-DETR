#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
python rtdetr_pytorch/tools/train.py \
    -c 'rtdetr_pytorch/configs/rtdetr/rtdetr_dual.yml' \
    -r 'output/1103_0/checkpoint_best.pth' \
    --test-only
#    > ../output/0410_0.log 2>&1 &

