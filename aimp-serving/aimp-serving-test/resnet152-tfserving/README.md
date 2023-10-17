# 介绍

Imagenet (ILSVRC-2012-CLS) classification with  Resnet152-V2 with input size 480x480.

### 算法介绍

1. 算法名称：resnet152-v2 （利用resnet152-v2 进行图像分类）
2. 输入：[1,480,480,3]的tensor, 图像为480x480大小，RGB3通道的图片， 1代表batch_size大小
3. 输出：1000维的向量，代表imagenet1000中的每一类的可能性的大小，具体类别名见 imagenet1000_clsidx_to_labels.txt

### 中台部署流程

1. 下载模型：https://tfhub.dev/google/imagenet/resnet_v2_152/classification/5

2. 打包zip文件,格式如下:

   ```
   │   resnet-v2-152-tfserving.zip
   │   ├──1
   │   │   ├── saved_model.pb
   │   │   ├── variables
   │   │   │   ├── variables.data-00000-of-00001
   │   │   │   ├── variables.index
   ```

3. 创建resnet-v2-152-tfserving.yaml文件

4. 运行run.py

   在run.py中，需要**更新infer_endpoint、infer-access-token、Host**三个变量，其需要在中台中获取，对应关系如下：
   infer_endpoint 为访问地址；infer-access-token为访问令牌；Host为目标服务。

   run.py运行成功将输出标签

   ```
   predict labels: cougar, puma, catamount, mountain lion, painter, panther, Felis concolor
   ```
