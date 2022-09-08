import io
import os
import time
import base64
import json
import time
import numpy as np 
import requests
from pprint import pprint
from transformers import BertTokenizer
import requests
import pickle 

# MUST import AIMP python SDK
# import upper dir's python file
import sys
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferWorkFlowSDK

#start init the aimpinferWorkFlowSDK
aimpPredict=aimpInferWorkFlowSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'resnet50-onnx-triton'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
infer_host_FQDN=aimpPredict.infer_host_FQDN
infer_endpoint=aimpPredict.infer_endpoint
#end init the aimpinferSDK

# step 4 predict
with open('./cat.pkl','rb') as f:
    img_data = pickle.load(f)
    
#import cv2
#path = './cat1.jpg'
#img = cv2.imread(path)
#img = cv2.resize(img,(224,224))
#img = img / 255.0
#img = np.expand_dims(img,axis = 3).astype(np.float32)
#img = np.transpose(img).astype('float32')
#print(img.shape)
#img = img.tolist()


data = {
   "inputs":[
   {
    "name": "input.1",
    "shape": [1,3,224,224],
    "datatype": "FP32",
    "data": img_data
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
r = requests.post(infer_endpoint, headers=headers, data=data, verify=False)
result = r.json()
pprint(result)

print('prediction probs:  ', result['outputs'][0]['data'])

with open('./imagenet1000_clsidx_to_labels.txt') as f:
    labels = eval(f.read())
print(labels[np.array(result['outputs'][0]['data']).argmax()])
