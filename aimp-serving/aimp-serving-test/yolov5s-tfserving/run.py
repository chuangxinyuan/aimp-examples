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
aimpPredict.model_name = 'yolov5s-tfserving'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
infer_host_FQDN=aimpPredict.infer_host_FQDN
infer_endpoint=aimpPredict.infer_endpoint
#end init the aimpinferSDK

import cv2


org_img = cv2.imread('./cat.jpg')
org_img = cv2.resize(org_img, (256,256))
img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img,axis=0).astype(np.float32)
img = img / 255.0

data = {
    'instances': img.tolist()
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
result = r.json()['predictions'][0]

parsed_ret = {}
num_det = result['output_3']
parsed_ret.update({'num_det': num_det})
parsed_ret.update({'boxes': np.array(result['output_0'][:num_det])*256})
parsed_ret.update({'probs': np.array(result['output_1'][:num_det])})
parsed_ret.update({'labels': np.array([int(i) for i in result['output_2'][:num_det]])})

print(parsed_ret)

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


for i in range(num_det):
    label = parsed_ret['labels'][i]
    x1, y1, x2, y2 = parsed_ret['boxes'][i]
    prob = parsed_ret['probs'][i]
    if prob > 0.3:
        cv2.rectangle(org_img, (int(x1),int(y1)),(int(x2),int(y2)), color=(255,0,0))
        cv2.putText(org_img, coco_names[label], (int(x1+10), int(x2+10)), 0, 0.75,(0,0,255))

cv2.imwrite('/mnt/output/cat_det.jpg',  org_img)

#result 说明
#output_1: prob
#output_0: xyxy boxes
#output_2 : cls_label
#output_3: num_detect
'''
{'predictions': [{'output_0': [[0.461833328, 0.587198, 0.551069736, 0.624527], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    'output_1': [0.427330375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'output_2': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'output_3': 1}]}
'''
