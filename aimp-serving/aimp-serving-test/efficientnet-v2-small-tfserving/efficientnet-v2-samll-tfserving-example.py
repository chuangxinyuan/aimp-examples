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
import pickle
from pprint import pprint

# MUST import AIMP python SDK
# import upper dir's python file
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferSDK

#start init the aimpinferSDK
aimpPredict=aimpInferSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'efficientnet-v2-small-tfserving'
aimpPredict.username='admin'
aimpPredict.password='admin'
#aimpPredict.token='5aed14f5bffc9f86fd0fb2745519f2ff'
aimpPredict.aimp_host='http://portal.aimpcloud.cn/api'
aimpPredict.infer_host='https://infer.aimpcloud.cn/'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
infer_host_FQDN=aimpPredict.infer_host_FQDN
infer_endpoint=aimpPredict.infer_endpoint
#end init the aimpinferSDK

# step 4 predict
#with open('./img.pkl','rb') as f:
    # img_data = pickle.load(f)

import cv2

path = './cat.jpg'
img = cv2.imread(path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(160,128))

img = img / 255.0
img = np.expand_dims(img,axis = 0).tolist()

data = {'instances': img}

headers = {
    'infer-access-token': access_token,
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
r = requests.post(infer_endpoint, headers=headers, data=json.dumps(data), verify=False)
result = r.json()
print('prediction probs: ', result['predictions'][0])

with open('imagenet1000_clsidx_to_labels.txt') as f:
    labels = eval(f.read())

print(labels[np.array(result['predictions'][0]).argmax()])
