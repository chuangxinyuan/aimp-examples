# 介绍
The Faster R-CNN model is based on the Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks paper.

# 模型准备方法
* Torch Serve模型准备
安装工具：
```bash 
pip install torch torch-vision
pip install torch-model-archiver
```
下载预训练模型，或训练模型：
``` bash 
cd pretrained_models
wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
```

* 打包成TorchServe Model Archive Files (MAR)--faster-rcnn-torchserve .mar
``` bash
torch-model-archiver --model-name faster-rcnn-torchserve --version 1.0 --model-file examples/object_detector/torch/faster-rcnn-pytorch/model.py --serialized-file pretrained_models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth --handler object_detector --extra-files examples/object_detector/torch/faster-rcnn-pytorch/index_to_name.json
```
* 模型准备
准备目录cd output
mkdir faster-rcnn-pytorch/config
mkdir faster-rcnn-pytorch/model-store
将前面生成的fasterrcnn.mar放入model-store文件夹下
在config文件夹下新建config.propertites, 文件内容如下：
``` bash
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
enable_metrics_api=true
metrics_format=prometheus
number_of_netty_threads=4
job_queue_size=10
service_envelope=kfserving
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"faster-rcnn-torchserve":{"1.0":{"defaultVersion":true,"marName":"faster-rcnn-torchserve.mar","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}

```
* 打包成zip文件, `zip faster-rcnn-torchserve.zip model-store config `
* 模型的结构和部署，请参考标准步骤

# 参考

https://pytorch.org/vision/stable/models/faster_rcnn.html

