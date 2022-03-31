_base_ = [
    '../multi_task_cdnet_vgg16_adam-lr5e-4_bs16_256x256_300e_monuseg_conf.py',
]

# model settings
model = dict(
    train_cfg=dict(
        num_angles=8,
        use_regression=False,
        noau=True,
        parallel=True,
        use_twobranch=False,
        use_distance=False,
        use_sigmoid=False,
        use_ac=True,
        ac_len_weight=1,
        use_focal=False,
        use_level=False,
        use_variance=True,
        use_tploss=False,
        tploss_weight=False,
        tploss_dice=False,
        dir_weight_map=True,
    ))
