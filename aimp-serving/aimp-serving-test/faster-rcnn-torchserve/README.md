# 介绍
The Faster R-CNN model is based on the Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks paper.

## 算法介绍

1. 算法名称：fasterrcnn（利用fasterrcnn进行目标检测）

2. 输入：RGB3通道的图片， 进行64位编码后的二进制流

3. 输出：列表类型，列表中每个元素（字典类型）代表一个检测结果，以示例图片cat.jpg为例，检测结果如下：

   ```python
   [{'cat': [101.26473236083984, 43.6788444519043, 321.4162902832031, 268.486572265625], 'score': 0.9853323101997375}, {'book': [126.29954528808594, 222.5118865966797, 414.14837646484375, 301.4258728027344], 'score': 0.7724350690841675}, {'book': [63.96335983276367, 221.37020874023438, 254.1805877685547, 300.2806701660156], 'score': 0.5295757055282593}]
   ```

   其中每个元素的字典类型key表示检测类型和检测概率。如第一个元素表示检测到了类别 cat，对应概率是0.985，检测框的位置（x1, y1, x2, y2) = [101.26473236083984, 43.6788444519043, 321.4162902832031, 268.486572265625]。其中x1,y1代表检测框的左上角的坐标。x2,y2表示右下角的坐标。

   类别名称参考coco数据集：https://cocodataset.org/

   注：检测结果可视化见cat_det.jpg

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

