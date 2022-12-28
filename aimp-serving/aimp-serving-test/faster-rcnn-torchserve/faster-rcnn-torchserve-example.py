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
import  cv2

# MUST import AIMP python SDK
# import upper dir's python file
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferSDK

#start init the aimpinferSDK
aimpPredict=aimpInferSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'faster-rcnn-torchserve'
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
headers = {
    'infer-access-token': access_token,
    'Content-Type': 'application/json',
    'Host': infer_host_FQDN,
}

print('---api_predict_endpoint and headers---')
print (infer_endpoint)
pprint(headers)
print('\n')

path = './cat.jpg'
image = open(path, 'rb') #open binary file in read mode
image_read = image.read()
image_64_encode = base64.b64encode(image_read)
bytes_array = image_64_encode.decode('utf-8')
data = {
    'instances': [
        {'data': bytes_array}
    ]
}

print('---Prediction RESULTS---')
# original predict URL
#r = requests.post(endpoint, headers=headers, data=json.dumps(data), verify=False)
# skip cert check
r = requests.post(infer_endpoint, headers=headers, data=json.dumps(data), verify=False)
result = r.json()

pic_result = result['predictions'][0] #the result of one picture

print(pic_result)

##screen result
list1 = []
for i in pic_result:
    if i['score'] > 0.5:
        list1.append(i)

#draw the frame
img = cv2.imread(path)

for ret in list1:
    for k, v in ret.items():
        if k == 'score':
            pass
        else:
            x1, y1, x2, y2 = [int(i) for i in v]
        
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
            img = cv2.putText(img,k,(x1-5,y1-10),0,1,(0,0,0),1)
cv2.imwrite('cat_det.jpg', img)
 


