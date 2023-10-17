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
from pprint import pprint

access_token="it-029f5933a3ff4bf88a1f5b10a0f26e31"
infer_host_FQDN="resnet-v2-152-tfserving.aimp.fat.aimpcloud.cn"
infer_endpoint="https://infer.fat.aimpcloud.cn/v1/models/resnet-v2-152-tfserving:predict"

import cv2
path = './cat.jpg'
img = cv2.imread(path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(480,480))

img = img / 255.0
img = np.expand_dims(img,axis = 0).tolist()

data = {'instances': img}


headers = {
    
    'infer-access-token': access_token,
    'Content-Type': 'application/json',
    'Host': infer_host_FQDN,
}

print('---Prediction RESULTS---')

r = requests.post(infer_endpoint, headers=headers, data=json.dumps(data), verify=False)
result = r.json()

with open('imagenet1000_clsidx_to_labels.txt') as f:
    labels = eval(f.read())
    
print("predict labels:", labels[np.array(result['predictions'][0]).argmax()])