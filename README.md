
# *<center>MRT-DETR: A Robust Visible-Infrared Object Detector with Adaptive Cross-modal Feature Fusion</center>*


## Requirements
- python 3.9
- torch 2.0.1
- torchvision 0.15.2

## Build
  ```
  cd rtdetr_pytorch
  pip install -r requirements.txt
  ```

## Datasets
 - Please ensure the dataset is downloaded and placed under your specified data path, such as /data/M3FD/.
 - The data structure should be as follows:
```
/data/M3FD/
├── annotations/
│   ├── train.json  # Training set annotations
│   └── val.json    # Validation set annotations
├── train_RGB/      # Training set Visible Light Images (RGB)
├── train_thermal/  # Training set Thermal Images (IR)
├── val_RGB/        # Validation set Visible Light Images (RGB)
└── val_thermal/    # Validation set Thermal Images (IR)
```

## Train
  ```
  sh run.sh
  ```

## Eval
  ```
  sh eval.sh
  ```

## Acknowledge
*This code is highly borrowed from [RT-DETR](https://github.com/lyuwenyu/RT-DETR). Thanks to Yian Zhao.

