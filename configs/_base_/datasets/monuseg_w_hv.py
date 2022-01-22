# dataset settings
dataset_type = 'NucleiMoNuSegDataset'
data_root = 'data/monuseg'
process_cfg = dict(
    if_flip=True,
    if_jitter=True,
    if_elastic=True,
    if_blur=True,
    if_crop=True,
    if_pad=True,
    if_norm=False,
    with_hv=True,
    min_size=256,
    max_size=2048,
    resize_mode='fix',
    edge_id=2,
)
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/c300',
        ann_dir='train/c300',
        split='only-train_t16_train_c300.txt',
        process_cfg=process_cfg),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/c0',
        ann_dir='train/c0',
        split='only-train_t16_test_c0.txt',
        process_cfg=process_cfg),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/c0',
        ann_dir='train/c0',
        split='only-train_t16_test_c0.txt',
        process_cfg=process_cfg),
)
