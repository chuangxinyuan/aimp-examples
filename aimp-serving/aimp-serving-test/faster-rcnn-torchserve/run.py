from __future__ import print_function

import io
import os
import time
import base64
import json
import time

import requests
import pickle
import onepanel.core.api
from onepanel.core.api.rest import ApiException
import onepanel.core.auth
from pprint import pprint
import sys

# MUST import AIMP python SDK
# import upper dir's python file
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferWorkFlowSDK

#start init the aimpinferWorkFlowSDK
aimpPredict=aimpInferWorkFlowSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'faster-rcnn-torchserve'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
infer_endpoint=aimpPredict.infer_endpoint
#end init the aimpinferSDK

# If inside of Onepanel, get mounted service account token to use as API Key
access_token = onepanel.core.auth.get_access_token()

import base64
image = open('./persons.jpg', 'rb') #open binary file in read mode
image_read = image.read()
image_64_encode = base64.b64encode(image_read)
bytes_array = image_64_encode.decode('utf-8')

data = {
    'instances': [
        {'data': bytes_array}
    ]
}

headers = {
    'onepanel-access-token': access_token,
    'Content-Type': 'application/json',
}

print('---api_predict_endpoint and headers---')
print (infer_endpoint)
pprint(headers)
print('\n')

print('---Prediction RESULTS---')
# original predict URL
r = requests.post(infer_endpoint, headers=headers, data=json.dumps(data), verify=False)
result = r.json()
pprint(result)
