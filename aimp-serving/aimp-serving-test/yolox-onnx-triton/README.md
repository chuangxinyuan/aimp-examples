# 介绍
YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models:
## 算法介绍
1. 算法名称：yolox （利用yolox进行目标检测）

2. 输入：[1,3,224,224]的tensor, 图像为224x224大小，BGR3通道的图片， 1代表batch_size大小

3. 输出：字典类型，包含dets和labels，以示例图片cat.jpg为例，检测结果如下：
   
    ```python
    {'dets': array([[ 12.61358643,  19.22767639, 159.27613831, 187.19633484,
            0.93422616],
        [  0.        ,   0.        ,   0.        ,   0.        ,
            0.        ]]),
    'labels': array([15,  0])}
    ```
    
    其中labels维度Nx1, N表示检测的物体个数。dets的维度Nx5, 5个维度（x1, y1, x2, y2, prob)。其中x1,y1代表检测框的左上角的坐标。x2,y2表示右下角的坐标。prob表示检测框的置信度。labels中是类别的索引，对应的类别名称见：
    
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
https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/02-how-to-run/convert_model.md
https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox/README.md
