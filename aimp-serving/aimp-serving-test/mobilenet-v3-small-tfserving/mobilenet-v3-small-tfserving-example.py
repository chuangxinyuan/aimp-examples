from __future__ import print_function
import io
import sys
import os

import json
import numpy as np 
import requests



# MUST import AIMP python SDK
# import upper dir's python file
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferSDK

#start init the aimpinferSDK
aimpPredict=aimpInferSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'mobilenet-v3-small-tfserving'
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
#with open('./img.pkl','rb') as f:
    #img_data = pickle.load(f)
    
import cv2
path = './cat.jpg'
img = cv2.imread(path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# For this model, the size of the input images is fixed to height x width = 224 x 224 pixels.
img = cv2.resize(img,(224,224))
img = img / 255.0
img_data = np.expand_dims(img,axis = 0).tolist()



data = {
    'instances': img_data
}

headers = {
    'onepanel-access-token': access_token,
    'Content-Type': 'application/json',
    'Host': infer_host_FQDN,
}

print('---api_predict_endpoint and headers---')
print (infer_endpoint)
print(headers)
print('\n')

print('---Prediction RESULTS---')
# original predict URL
#r = requests.post(endpoint, headers=headers, data=json.dumps(data), verify=False)
# skip cert check
r = requests.post(infer_endpoint, headers=headers, data=json.dumps(data), verify=False)
result = r.json()
print('features: ', result['predictions'][0])


