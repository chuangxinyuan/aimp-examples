from __future__ import print_function
import io
import sys
import os
import time
import base64
import json
import time
import numpy as np 
import requests
from pprint import pprint
import pickle

# MUST import AIMP python SDK
# import upper dir's python file
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferSDK

#start init the aimpinferSDK
aimpPredict=aimpInferSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'yolox-onnx-triton'
aimpPredict.username='admin'
aimpPredict.token='5aed14f5bffc9f86fd0fb2745519f2ff'
aimpPredict.aimp_host='http://onepanel.niuhongxing.cn/api'
aimpPredict.infer_host='https://infer.dev.aimpcloud.cn/'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
infer_host_FQDN=aimpPredict.infer_host_FQDN
infer_endpoint=aimpPredict.infer_endpoint
#end init the aimpinferSDK

# step 4 predict
headers = {
    'onepanel-access-token': access_token,
    'Content-Type': 'application/json',
    'Host': infer_host_FQDN,
}

print('---api_predict_endpoint and headers---')
print (infer_endpoint)
pprint(headers)
print('\n')

with open('./img.pkl','rb') as f:
    # shape of img_data: [1,3, 224, 224]
    img_data = pickle.load(f).astype('float32').tolist()

data = {
   "inputs":[
   {
    "name": "input",
    "shape": [1,3,224,224],
    "datatype": "FP32",
    "data": img_data
   }
   ]
}

print('---Prediction RESULTS---')
# original predict URL
#r = requests.post(endpoint, headers=headers, data=json.dumps(data), verify=False)
# skip cert check
r = requests.post(infer_endpoint, headers=headers, data=json.dumps(data), verify=False)
result = r.json()
pprint(result)

 
#result 说明

#labels: cls_label
#dets: xyxy boxes with the prob (flattened)
'''
{'model_name': 'yolox-onnx-triton', 'model_version': '1', 'outputs':
[{'name': 'dets', 'datatype': 'FP32', 'shape': [1, 3, 5], 'data': [126.69447326660156, 76.2667465209961, 180.7909698486328, 177.94143676757812, 0.887129545211792, 60.801849365234375, 24.55431365966797, 106.42037963867188, 202.824951171875, 0.8643292188644409, 0.0, 0.0, 0.0, 0.0, 0.0]},
 {'name': 'labels', 'datatype': 'INT64', 'shape': [1, 3], 'data': [0, 0, 0]}]}
'''
# cls_label 到 cls_name 可以从下面的coco128_names进行查询
coco128_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names



