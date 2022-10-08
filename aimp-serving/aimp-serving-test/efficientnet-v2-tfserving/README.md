# 介绍
Imagenet (ILSVRC-2012-CLS) classification with EfficientNet V2 with input size 480x480.

## 算法介绍

1. 算法名称：efficientnet-v2（利用efficientnet-v2进行图像分类）
2. 输入：[1,480,480,3]的tensor, 图像为480x480大小，RGB3通道的图片， 1代表batch_size大小
3. 输出：1000维的向量，代表imagenet1000中的每一类的可能性的大小，具体类别名见  imagenet1000_clsidx_to_labels.txt

# 参考

https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2

# 模型准备方法

### 下载预训练模型并解压

- wget https://storage.googleapis.com/tfhub-modules/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2.tar.gz
- tar -zxvf 2.tar.gz

### 打包成TFS需要的目录格式

```
│   efficientnet-v2-tfserving.zip
│   ├──1
│   │   ├── savedmodel.pb
│   │   ├── variables
│   │   │   ├── variables.data-00000-of-00001
│   │   │   ├── variables.index
```



