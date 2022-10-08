# 介绍
The ResNet model is based on the Deep Residual Learning for Image Recognition paper.
# 参考
https://pytorch.org/vision/stable/models.html

## 算法介绍
1. 算法名称：resnet50 （利用restnet50进行图像分类）
2. 输入：[1,3,224,224]的tensor, 图像为224x224大小，RGB3通道的图片， 1代表batch_size大小
3. 输出：1000维的向量，代表imagenet1000中的每一类的可能性的大小，具体类别名见  imagenet1000_clsidx_to_labels.txt

# 模型准备方法

### 下载预训练模型，或训练模型并且转换为onnx 模型

```python
import torch
import torchvision
device = torch.device("cpu")
net = torchvision.models.resnet50(pretrained=True)
net = net.to(device)
net.eval()
# print(net)
tmp = torch.ones(1, 3, 224, 224).to(device)
out = net(tmp)
print('resnet50 out:', out.shape)
torch.save(net,'resnet50.pth')
#通过torch.onnx.export转化成onnx模型文件
model = torch.load("resnet50.pth")
inputs = torch.randn(1, 3, 224, 224)
device = torch.device("cpu")
inputs = inputs.to(device)
torch.onnx.export(model, inputs, '/Users/jingyi_wu/Desktop/resnet50.onnx', export_params=True, verbose=True)

```

###  打包成triton需要的目录格式

```
├──resnet50-onnx-triton.zip
│   ├──resnet50-onnx-triton
│   │   ├── config.pbtxt
│   │   ├── 1
│   │   │   ├── model.onnx
```



- 其中config.pbtxt是triton server的配置文件

```

name: "resnet50-onnx-triton"
platform: "onnxruntime_onnx"
input [
{
name: "input.1"
data_type: TYPE_FP32
dims: [1,3,224,224 ]
}]
output [
{
name: "495"
data_type: TYPE_FP32
dims: [ 1, 1000 ]
}
]
 
version_policy: { latest { num_versions : 1 }}
optimization {
graph { level: 1 }
}
```