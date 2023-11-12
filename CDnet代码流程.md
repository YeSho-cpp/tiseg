### 训练配置部分

我们可以在[这里](configs/cdnet/monuseg_dir.py)配置进程数、batchsize、数据路径以及数据增强方法，在[这里](configs/cdnet/cdnet_vgg16_adam-lr5e-4_bs16_256x256_300e_monuseg.py)配置一些基础的超参数(学习率、权重衰减、迭代次数等等)

### 模型网络部分

这个代码的模型是分为
```sh
├── backbones
├── heads
├── losses
├── segmentors
```
这些模块

每个方法的segmentors都要继承一个基础的[segmentors](tiseg/models/segmentors/base.py#L50)

CDnet的segmentors在[这里](tiseg/models/segmentors/cdnet.py)

每个方法的heads继承一个[UNetHead](tiseg/models/heads/unet_head.py#L52)



一些重要参数设置
- CDnet的方向角度数量在[这里](tiseg/models/segmentors/cdnet.py#L27)




### 网络图

<img src="https://article.biliimg.com/bfs/article/a41714d77798126c92f37481d847c51a38716159.png" alt="image.png" style="zoom:70%;" />


DGM模块的代码在CDHead[这里](tiseg/models/heads/cd_head.py#L136)

点分支、方向分支和掩码分支运算在[这里](iseg/models/heads/cd_head.py#96)定义

三个分支的注意力单元AU在[这里](iseg/models/heads/cd_head.py#L102)定义



这是模型的详细信息

```sh
train_processes = [
    dict(
        type='Affine',
        scale=(0.8, 1.2),
        shear=5,
        rotate_degree=[-180, 180],
        translate_frac=(0, 0.01)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomCrop', crop_size=(256, 256)),
    dict(type='Pad', pad_size=(256, 256)),
    dict(type='RandomBlur'),
    dict(
        type='ColorJitter',
        hue_delta=8,
        saturation_range=(0.8, 1.2),
        brightness_delta=26,
        contrast_range=(0.75, 1.25)),
    dict(
        type='Normalize',
        mean=[0.68861804, 0.46102882, 0.61138992],
        std=[0.19204499, 0.20979484, 0.1658672],
        if_zscore=False),
    dict(type='BoundLabelMake', edge_id=2, selem_radius=(3, 3)),
    dict(type='DirectionLabelMake'),
    dict(
        type='Formatting',
        data_keys=['img'],
        label_keys=[
            'sem_gt', 'sem_gt_w_bound', 'dir_gt', 'point_gt', 'loss_weight_map'
        ])
]
test_processes = [
    dict(
        type='Normalize',
        mean=[0.68861804, 0.46102882, 0.61138992],
        std=[0.19204499, 0.20979484, 0.1658672],
        if_zscore=False),
    dict(type='Formatting', data_keys=['img'], label_keys=[])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type='MoNuSegDataset',
        data_root='data/monuseg',
        img_dir='train/w512_s256',
        ann_dir='train/w512_s256',
        split='only-train_t12_v4_train_w512_s256.txt',
        processes=[
            dict(
                type='Affine',
                scale=(0.8, 1.2),
                shear=5,
                rotate_degree=[-180, 180],
                translate_frac=(0, 0.01)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(type='RandomCrop', crop_size=(256, 256)),
            dict(type='Pad', pad_size=(256, 256)),
            dict(type='RandomBlur'),
            dict(
                type='ColorJitter',
                hue_delta=8,
                saturation_range=(0.8, 1.2),
                brightness_delta=26,
                contrast_range=(0.75, 1.25)),
            dict(
                type='Normalize',
                mean=[0.68861804, 0.46102882, 0.61138992],
                std=[0.19204499, 0.20979484, 0.1658672],
                if_zscore=False),
            dict(type='BoundLabelMake', edge_id=2, selem_radius=(3, 3)),
            dict(type='DirectionLabelMake'),
            dict(
                type='Formatting',
                data_keys=['img'],
                label_keys=[
                    'sem_gt', 'sem_gt_w_bound', 'dir_gt', 'point_gt',
                    'loss_weight_map'
                ])
        ]),
    val=dict(
        type='MoNuSegDataset',
        data_root='data/monuseg',
        img_dir='val/w0_s0',
        ann_dir='val/w0_s0',
        split='only-train_t12_v4_val_w0_s0.txt',
        processes=[
            dict(
                type='Normalize',
                mean=[0.68861804, 0.46102882, 0.61138992],
                std=[0.19204499, 0.20979484, 0.1658672],
                if_zscore=False),
            dict(type='Formatting', data_keys=['img'], label_keys=[])
        ]),
    test=dict(
        type='MoNuSegDataset',
        data_root='data/monuseg',
        img_dir='test/w0_s0',
        ann_dir='test/w0_s0',
        split='only-train_t12_v4_test_w0_s0.txt',
        processes=[
            dict(
                type='Normalize',
                mean=[0.68861804, 0.46102882, 0.61138992],
                std=[0.19204499, 0.20979484, 0.1658672],
                if_zscore=False),
            dict(type='Formatting', data_keys=['img'], label_keys=[])
        ]))
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
runner = dict(type='EpochBasedRunner', max_epochs=300)
evaluation = dict(
    interval=30,
    custom_intervals=[1],
    custom_milestones=[295],
    by_epoch=True,
    metric='all',
    save_best='mAji',
    rule='greater')
checkpoint_config = dict(by_epoch=True, interval=5, max_keep_ckpts=5)
optimizer = dict(type='RAdam', lr=0.0002, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='step',
    by_epoch=True,
    step=[200],
    gamma=0.1,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1e-06)
model = dict(
    type='CDNet',
    num_classes=2,
    train_cfg=dict(),
    test_cfg=dict(
        mode='split',
        radius=3,
        if_ddm=True,
        crop_size=(256, 256),
        overlap_size=(40, 40),
        rotate_degrees=[0, 90],
        flip_directions=['none', 'horizontal', 'vertical', 'diagonal']))
work_dir = './work_dirs/cdnet/cdnet_vgg16_adam-lr5e-4_bs16_256x256_300e_monuseg'
gpu_ids = range(0, 1)

2023-11-11 21:45:12,536 - TorchImageSeg - INFO - Set random seed to 2086550877, deterministic: False
tools/train.py:126: UserWarning: SyncBN only support DDP. In order to compat with DP, we convert SyncBN tp BN. Please to use dist_train.py which has official support to avoid this problem.
  warnings.warn('SyncBN only support DDP. In order to compat with DP, we convert '
2023-11-11 22:32:53,365 - TorchImageSeg - INFO - CDNet(
  (backbone): TorchVGG16BN(
    (stages): ModuleList(
      (0): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (9): ReLU(inplace=True)
      )
      (3): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (9): ReLU(inplace=True)
      )
      (4): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (5): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (9): ReLU(inplace=True)
      )
      (5): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
    )
  )
  (head): CDHead(
    (decode_layers): ModuleList(
      (0): UNetLayer(
        (up_conv): Sequential(
          (0): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convs): Sequential(
          (0): ConvModule(
            (conv): Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
      )
      (1): UNetLayer(
        (up_conv): Sequential(
          (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convs): Sequential(
          (0): ConvModule(
            (conv): Conv2d(640, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
      )
      (2): UNetLayer(
        (up_conv): Sequential(
          (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convs): Sequential(
          (0): ConvModule(
            (conv): Conv2d(320, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
      )
      (3): UNetLayer(
        (up_conv): Sequential(
          (0): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convs): Sequential(
          (0): ConvModule(
            (conv): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
      )
      (4): UNetLayer(
        (up_conv): Sequential(
          (0): ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (convs): Sequential(
          (0): ConvModule(
            (conv): Conv2d(80, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(inplace=True)
          )
        )
      )
    )
    (postprocess): DGM(
      (mask_feats): RU(
        (act_layer): ReLU(inplace=True)
        (residual_ops): Sequential(
          (0): ConvModule(
            (conv): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ReLU(inplace=True)
          (2): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (identity_ops): Sequential(
          (0): ConvModule(
            (conv): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      )
      (dir_feats): RU(
        (act_layer): ReLU(inplace=True)
        (residual_ops): Sequential(
          (0): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ReLU(inplace=True)
          (2): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (identity_ops): Sequential(
          (0): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      )
      (point_feats): RU(
        (act_layer): ReLU(inplace=True)
        (residual_ops): Sequential(
          (0): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): ReLU(inplace=True)
          (2): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (identity_ops): Sequential(
          (0): ConvModule(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      )
      (point_to_dir_attn): AU(
        (conv): Sequential(
          (0): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): Sigmoid()
        )
      )
      (dir_to_mask_attn): AU(
        (conv): Sequential(
          (0): Conv2d(9, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): Sigmoid()
        )
      )
      (point_conv): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
      (dir_conv): Conv2d(64, 9, kernel_size=(1, 1), stride=(1, 1))
      (mask_conv): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)
```



### 数据加载

每个数据加载方法都要继承一个基本的[数据处理](tiseg/datasets/custom.py#L108)

CDNet的数据加载在[这里](tiseg/datasets/monuseg.py)

数据的类别通过这种方式加载
```python
  def __init__(self, **kwargs):
      super().__init__(img_suffix='.tif', sem_suffix='_sem.png', inst_suffix='_inst.npy', **kwargs)
```

加载数据增强的各种方法在[这里](tiseg/datasets/custom.py#L133)
数据增强的方法种类在[这里](tiseg/datasets/ops/__init__.py#L18)
其中CDnet的三个分支的生成在[DirectionLabelMake](/share/home/ncu10/Code/AI/NucleiSeg/tiseg/tiseg/datasets/ops/direction_map.py#L11)这个方法

### 训练流程

训练流程在[这里](tiseg/apis/train.py#L66)开始

1. 先用数据加载器加载之前的处理的数据
2. 构建一个运行器，并注册训练钩子(Hooks)
3. 开始调用各种数据增强的方法处理数据
4. 进入[训练步骤](tiseg/models/segmentors/base.py#L70),CDNet的foward在[这里](tiseg/models/segmentors/cdnet.py#L51)