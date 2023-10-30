# 数据处理命令
python tools/convert_dataset/monuseg.py data/monuseg only-train_t12_v4 -w 512 -s 256

# 训练命令
python tools/train.py configs/unet/unet_vgg16_adam-lr1e-4_bs8_256x256_300e_monuseg.py

# 测试命令

python tools/test.py configs/cdnet/cdnet_vgg16_adam-lr5e-4_bs16_256x256_300e_monuseg.py work_dirs/cdnet/cdnet_vgg16_adam-lr5e-4_bs16_256x256_300e_monuseg/best_mAji_epoch_160.pth
