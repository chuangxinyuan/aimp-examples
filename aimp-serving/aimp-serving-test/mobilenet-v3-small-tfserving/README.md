# 介绍
Feature vectors of images with MobileNet V3 small(depth multiplier 1.00) trained on ImageNet (ILSVRC-2012-CLS).

## 算法介绍

1. 算法名称：mobilenet-v3-small（利用mobilenet-v3-small进行图像特征提取）
2. 输入：[1,224,224,3]的tensor, 图像为224x224大小，RGB3通道的图片， 1代表batch_size大小
3. 输出：1024维的向量，代表对输入的图像经过神经网络后所提取的特征向量

# 参考
https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5
