# dataset settings
dataset_type = 'NucleiCoNICDatasetWithDirection'
data_root = 'data/conic'
process_cfg = dict(
    if_flip=True,
    if_jitter=True,
    if_elastic=True,
    if_blur=True,
    if_crop=True,
    if_pad=True,
    if_norm=False,
    with_dir=True,
    test_with_dir=True,
    min_size=256,
    max_size=2048,
    resize_mode='fix',
    edge_id=7,
    to_center=True,
    num_angles=8,
)
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/',
        ann_dir='train/',
        split='train.txt',
        process_cfg=process_cfg),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/',
        ann_dir='val/',
        split='val.txt',
        process_cfg=process_cfg),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/',
        ann_dir='val/',
        split='val.txt',
        process_cfg=process_cfg),
)
