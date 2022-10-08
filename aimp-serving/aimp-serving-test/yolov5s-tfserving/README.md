# 介绍
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

## 算法介绍

1. 算法名称：yolov5s （利用yolov5s进行目标检测）

2. 输入：[1,256, 256, 3]的tensor, 图像为256x256大小，RGB3通道的图片， 1代表batch_size大小

3. 输出：字典类型，包含num_det, boxes, probs和labels，以示例图片cat.jpg为例，检测结果如下：

   ```python
   {'num_det': 2, 'boxes': array([[ 14.36322785,  27.23092659, 183.25946035, 210.49017344],
          [ 34.253952  , 174.675968  , 243.76280218, 243.9891968 ]]), 'probs': array([0.89651293, 0.53926593]), 'labels': array([15, 73])}
   ```

   其中num_det表示检测到的物体数量。labels维度num_det x 1。boxes的维度num_det x 4, 4个维度（x1, y1, x2, y2,)。其中x1,y1代表检测框的左上角的坐标。x2,y2表示右下角的坐标。prob表示检测框的置信度。labels中是类别的索引，对应的类别名称见：

   ```python
   coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush'] 
   ```

   注：检测结果可视化见cat_det.jpg

# 参考
https://github.com/ultralytics/yolov5/issues/251
https://github.com/ultralytics/yolov5