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

with open('./img.pkl','rb') as f:
    # shape of img_data: [1, 256, 256, 3]
    img_data = pickle.load(f)
    img_data = img_data/255.0

data = {
    'instances': img_data.tolist()
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
result = r.json()
pprint(result)


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
