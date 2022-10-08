# 介绍
Imagenet (ILSVRC-2012-CLS) classification with EfficientNet V2 with input size 480x480.

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



