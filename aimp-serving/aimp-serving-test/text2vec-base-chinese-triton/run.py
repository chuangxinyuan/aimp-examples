import os
import time
import base64
import json
import time
import requests
import onepanel.core.api
from onepanel.core.api.rest import ApiException
import onepanel.core.auth
from transformers import BertTokenizer
import requests

# MUST import AIMP python SDK
# import upper dir's python file
import sys
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferWorkFlowSDK
import numpy as np 
#start init the aimpinferWorkFlowSDK
aimpPredict=aimpInferWorkFlowSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'text2vec-base-chinese-triton'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
endpoint=aimpPredict.infer_endpoint
#end init the aimpinferSDK

tokenizer = BertTokenizer.from_pretrained("shibing624/text2vec-base-chinese")
inputs = tokenizer(["如何更换花呗绑定银行卡"]) 


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
    'onepanel-access-token': access_token
}

r = requests.post(endpoint,headers=headers, json=data)

result = r.json()

print('sentence embeddings:  ', result['outputs'][0]['data'])
