# 自动标注模型
自动标注模型服务运行在AIMP中，可以被CVAT 1.6 自动标注功能使用。
## 模型定义文件编写
有如下关键字段，请参见示例模型定义文件：
1. metadata.label "used-by", cvat会自动显示拥有该标签的模型
1. metadata.annotations.spec是模型的预测分类，应用在cvat中，会在cvat选择自动标注时，显示在弹出的界面上
1. metadata.annotations.modelName，是cvat中使用该模型的名字，可以和模型的名字不一致，应用在cvat中，会在cvat选择自动标注时，显示在弹出的界面上
1. metadata.annotations.framework: "tensorflow"， 请照抄该字段
1. metadata.annotations.type: detector， 请照抄该字段

## 自动标注模型示例：KFServing MASK RCNN
### 介绍
This is a proof-of-concept imitation of a MASK RCNN model using tensorflow, implemented using kfserving.

The inspiration comes from CVAT's nuclio model

https://github.com/openvinotoolkit/cvat/tree/develop/serverless/tensorflow/matterport/mask_rcnn/nuclio


### Building kerasserver
Using docker, run

```bash
docker build -t <some-name:some-tag> . -f keras.Dockerfile 
```
### 使用
1. kerasRcnnModel.yaml可以在cvat1.6 内置工作区中运行，作用是图像分割自动标注
2. 该模型部署到AIMP中，注意namespace字段要替换成自己的的namespace
