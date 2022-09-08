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
import aimpInferWorkFlowSDK

#start init the aimpinferSDK
aimpPredict=aimpInferWorkFlowSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'mobilenet-v3-small-tfserving'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
infer_host_FQDN=aimpPredict.infer_host_FQDN
infer_endpoint=aimpPredict.infer_endpoint
#end init the aimpinferSDK

# step 4 predict
with open('./cat.pkl','rb') as f:
    img_data = pickle.load(f)
    
#import cv2
#import matplotlib.pyplot as plt
#path = './cat1.jpg'
#img = cv2.imread(path)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#img = cv2.resize(img,(224,224))
#plt.imshow(img)
#img = img / 255.0
#img_data = np.expand_dims(img,axis = 0).tolist()

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
pprint(headers)
print('\n')
print('---Prediction RESULTS---')
# original predict URL
#r = requests.post(endpoint, headers=headers, data=data, verify=False)
# skip cert check
r = requests.post(infer_endpoint, headers=headers, data=data, verify=False)
result = r.json()
pprint(result)
