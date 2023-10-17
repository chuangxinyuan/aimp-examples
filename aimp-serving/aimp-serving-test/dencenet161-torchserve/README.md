# 介绍

使用torchserve进行图像分类

### 算法介绍

1. 算法名称：densenet-161 （利用densenet-161 进行图像分类）
2. 输入：图片
3. 输出：类别

### 中台部署流程

1. 下载模型：wget https://download.pytorch.org/models/densenet161-8d451a50.pth

2. 使用torch-model-archiver 生成.mar文件

   ```
   torch-model-archiver --model-name densenet161-torchserve --version 1.0 --model-file ./model.py --serialized-file densenet161-8d451a50.pth --handler image_classifier --extra-files ./index_to_name.json
   ```

3. 打包zip文件,格式如下:

   ```
   │   densenet161-torchserve.zip
   │   ├── config
   │   │   ├── config.properties
   │   ├── model-store
   │   │   │   ├── densenet161-torchserve.mar
   ```

4. 创建densenet161-torchserve.yaml文件

5. 运行run.py

   在run.py中，需要**更新infer_endpoint、infer-access-token、Host**三个变量，其需要在中台中获取，对应关系如下：
   infer_endpoint 为访问地址；infer-access-token为访问令牌；Host为目标服务。

   run.py运行成功将输出标签

   ```
   ---Prediction RESULTS---
   predict label: tiger_cat
   ```

### 参考

- [serve/examples/image_classifier/densenet_161/README.md at master · pytorch/serve (github.com)](https://github.com/pytorch/serve/blob/master/examples/image_classifier/densenet_161/README.md)

