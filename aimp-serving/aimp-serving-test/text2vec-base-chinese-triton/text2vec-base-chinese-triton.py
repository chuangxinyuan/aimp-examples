from __future__ import print_function
import io
import sys
import os
import time
import base64
import json
import time
import requests
from pprint import pprint
from transformers import BertTokenizer
import numpy as np 
# MUST import AIMP python SDK
# import upper dir's python file
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferSDK

#start init the aimpinferSDK
aimpPredict=aimpInferSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'text2vec-base-chinese-triton'
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
tokenizer = BertTokenizer.from_pretrained("shibing624/text2vec-base-chinese")
inputs_txt = ["如何更换花呗绑定银行卡"]
inputs = tokenizer(inputs_txt) 

data = {
   "inputs":[
   {
    "name": "input_ids",
    "shape": np.array(inputs['input_ids']).shape,
    "datatype": "INT32",
    "data": inputs['input_ids']
   },
    {
    "name": "attention_mask",
    "shape": np.array(inputs['attention_mask']).shape,
    "datatype": "INT32",
    "data": inputs['attention_mask']
   }, 
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
r = requests.post(infer_endpoint, headers=headers, data=json.dumps(data), verify=False)
result = r.json()

print('inputs sentence: ', inputs_txt)
print('sentence embeddings:  ', result['outputs'][0]['data'])
print(np.array(result['outputs'][0]['data']).shape)

