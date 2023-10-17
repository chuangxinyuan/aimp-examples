# 介绍

使用torchserve进行图像分割

### 算法介绍

1. 算法名称：deeplabv3-resnet-101 （利用deeplabv3-resnet-101 进行图像分类）
2. 输入：图片
3. 输出：分割后的图像和像素点的类别和概率

### 中台部署流程

1. 下载模型：https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth

2. 使用torch-model-archiver 生成.mar文件

   ```
   torch-model-archiver --model-name deeplabv3_resnet_101 --version 1.0 --model-file ./model.py --serialized-file deeplabv3_resnet101_coco-586e9e4e.pth --handler image_segmenter --extra-files ./deeplabv3.py,./intermediate_layer_getter.py,./fcn.py
   ```

3. 打包zip文件,格式如下:

   ```
   │   deeplabv3-resnet-101-torchserve.zip
   │   ├── config
   │   │   ├── config.properties
   │   ├── model-store
   │   │   │   ├── deeplabv3-resnet-101-torchserve.mar
   ```

4. 创建densenet161-torchserve.yaml文件

5. 运行run.py

   在run.py中，需要**更新infer_endpoint、infer-access-token、Host**三个变量，其需要在中台中获取，对应关系如下：
   infer_endpoint 为访问地址；infer-access-token为访问令牌；Host为目标服务。

   run.py运行成功将输出以下内容

   ![image-20231010170125178](C:\Users\liuyinglai\AppData\Roaming\Typora\typora-user-images\image-20231010170125178.png)
   
   ```
   ---Prediction RESULTS---
   [[[0.0, 0.9988763332366943], [0.0, 0.9988763332366943], [0.0, 0.9988763332366943], [0.0, 0.9988763332366943], [0.0, 0.9988666772842407], [0.0, 0.9988440275192261], [0.0, 0.9988170862197876], [0.0, 0.9987859725952148] ... ]]
   ```

### 参考

- [serve/examples/image_segmenter/deeplabv3/README.md at master · pytorch/serve (github.com)](https://github.com/pytorch/serve/blob/master/examples/image_segmenter/deeplabv3/README.md)

