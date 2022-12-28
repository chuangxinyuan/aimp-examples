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
aimpPredict.model_name = 'yelp-polarity-triton'
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
tokenizer = BertTokenizer.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")
inputs_txt = ["Although this movie is fantastic,and my daughter likes it, I still hate it."]
#inputs = tokenizer(["Hello, my dog is cute"]) #pos
#inputs = tokenizer(["This movie is fantastic."]) #pos
#inputs = tokenizer(["This movie is fantastic,and my daughter likes it."]) #pos
inputs = tokenizer(inputs_txt) #neg
#inputs = tokenizer(["i hate you "]) #neg

data = {
   "inputs":[
   {
    "name": "input_0",
    "shape": [1,100],
    "datatype": "INT32",
    "data": inputs['input_ids']
   }
   ]
}

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

print('inputs text: ', inputs_txt)
print('prediction probs:  ', result['outputs'][0]['data'])

labels = ['negative','positive']

print('prediction label: ', labels[np.array(result['outputs'][0]['data']).argmax()])
