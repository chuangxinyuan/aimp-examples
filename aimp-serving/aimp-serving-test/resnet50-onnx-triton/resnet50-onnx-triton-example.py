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
from transformers import BertTokenizer

# MUST import AIMP python SDK
# import upper dir's python file
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferSDK

#start init the aimpinferSDK
aimpPredict=aimpInferSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'resnet50-onnx-triton'
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
import cv2
path = './cat.jpg'
img = cv2.imread(path)
img = cv2.resize(img,(224,224))
img = img / 255.0
img = np.expand_dims(img,axis = 3).astype(np.float32)
img = np.transpose(img).astype('float32')
#print(img.shape)
img = img.tolist()


data = {
   "inputs":[
   {
    "name": "input.1",
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
#r = requests.post(endpoint, headers=headers, data=json.dumps(data), verify=False)
# skip cert check
r = requests.post(infer_endpoint, headers=headers, json=data)
result = r.json()

print('prediction probs:  ', result['outputs'][0]['data'])

with open('./imagenet1000_clsidx_to_labels.txt') as f:
    labels = eval(f.read())
print(labels[np.array(result['outputs'][0]['data']).argmax()])
