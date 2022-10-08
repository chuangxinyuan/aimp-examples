from __future__ import print_function

import io
import os
import time
import base64
import json
import time
import numpy as np 
import requests
import pickle
import onepanel.core.api
from onepanel.core.api.rest import ApiException
import onepanel.core.auth
from pprint import pprint

# MUST import AIMP python SDK
# import upper dir's python file
import sys
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferWorkFlowSDK

#start init the aimpinferWorkFlowSDK
aimpPredict=aimpInferWorkFlowSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'yolox-onnx-triton'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
infer_host_FQDN=aimpPredict.infer_host_FQDN
infer_endpoint=aimpPredict.infer_endpoint
#end init the aimpinferSDK

# step 4 predict
import cv2

org_img = cv2.imread('./cat.jpg')
org_img = cv2.resize(org_img, (224,224))
img = np.expand_dims(org_img,axis=0).astype(np.float32)
img = np.transpose(img, (0,3,1,2))

img = img.tolist()

data = {
   "inputs":[
   {
    "name": "input",
    "shape": [1,3,224,224],
    "datatype": "FP32",
    "data": img
   }
   ]
}


headers = {
    'onepanel-access-token': access_token,
    'Content-Type': 'application/json',
    'Host': infer_host_FQDN,
}
print('---api_predict_endpoint and headers---')
print (infer_endpoint)
pprint(headers)
print('\n')
print('---Prediction RESULTS---')
# original predict URL
#r = requests.post(endpoint, headers=headers, data=data, verify=False)
# skip cert check
r = requests.post(infer_endpoint, headers=headers, data=json.dumps(data), verify=False)

result = r.json()['outputs']
parsed_ret = {}
for ret in result:
    parsed_ret.update({ret['name']:np.array(ret['data']).reshape(ret['shape'][1:])})
pprint(parsed_ret)

# cls_label 到 cls_name 可以从下面的coco_names进行查询
coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names

dets = parsed_ret['dets']
labels = parsed_ret['labels']
for i in range(len(labels)):
    label = labels[i]
    x1, y1, x2, y2, prob = dets[i]
    if prob > 0.3:
        cv2.rectangle(org_img, (int(x1),int(y1)),(int(x2),int(y2)), color=(255,0,0))
        cv2.putText(org_img, coco_names[label], (int(x1+10), int(x2+10)), 0, 0.75,(0,0,255))

cv2.imwrite('./cat_det.jpg',  org_img)
 
#result 说明

#labels: cls_label
#dets: xyxy boxes with the prob (flattened)
'''
{'model_name': 'yolox-onnx-triton', 'model_version': '1', 'outputs':
[{'name': 'dets', 'datatype': 'FP32', 'shape': [1, 3, 5], 'data': [126.69447326660156, 76.2667465209961, 180.7909698486328, 177.94143676757812, 0.887129545211792, 60.801849365234375, 24.55431365966797, 106.42037963867188, 202.824951171875, 0.8643292188644409, 0.0, 0.0, 0.0, 0.0, 0.0]},
 {'name': 'labels', 'datatype': 'INT64', 'shape': [1, 3], 'data': [0, 0, 0]}]}
'''

