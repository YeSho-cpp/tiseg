# Tissue-Image-Segmentation

unofficial re-implementation of popular tissue image segmentation models

Support Model:

- [x] UNet
- [x] Dist
- [x] DCAN
- [x] MicroNet
- [x] FullNet
- [x] CDNet

## Dataset Prepare

Please check [this doc](docs/data_prepare.md)

Supported Dataset:

- [x] MoNuSeg;
- [x] CoNSeP;
- [x] CPM17;
- [x] CoNIC;

## Installation

1. Install MMCV-full (Linux recommend): `pip install MMCV-full==1.3.13`;
2. Install requirements package: `pip install -r requirements.txt`;
3. Install tiseg: `pip install -e .`;

## Usage

### Training

```Bash
# single gpu training
python tools/train.py [config_path]
# multiple gpu training
./tools/dist_train.sh [config_path] [num_gpu]
# 用unet训练monuseg
python tools/train.py configs/unet/unet_vgg16_adam-lr1e-4_bs8_256x256_300e_monuseg.py
# 用 4gpu unet训练monuseg
python tools/train.py configs/unet/unet_vgg16_adam-lr1e-4_bs8_256x256_300e_monuseg.py 4
```

# Evaluation

```Bash

# single gpu evaluation
python tools/test.py [config_path] [checkpoint] [--save-pred] # 带上--save-pred表示保存预测的图片结果
# multiple gpu evaluation
python ./tools/dist_test.py [config_path] [checkpoint] [--save-pred] [num_gpu]

# 验证cdnet的测试结果 
python tools/test.py configs/cdnet/cdnet_vgg16_adam-lr5e-4_bs16_256x256_300e_monuseg.py work_dirs/cdnet/cdnet_vgg16_adam-lr5e-4_bs16_256x256_300e_monuseg/best_mAji_epoch_160.pth

# 验证cdnet的测试结果 (保存预测图片)
python tools/test.py configs/cdnet/cdnet_vgg16_adam-lr5e-4_bs16_256x256_300e_monuseg.py work_dirs/cdnet/cdnet_vgg16_adam-lr5e-4_bs16_256x256_300e_monuseg/best_mAji_epoch_160.pth --save-pred
```

## Thanks

This repo follow the design mode of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) & [detectron2](https://github.com/facebookresearch/detectron2).
