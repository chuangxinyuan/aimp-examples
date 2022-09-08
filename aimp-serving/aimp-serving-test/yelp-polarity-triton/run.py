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
import numpy as np 
from pprint import pprint

# MUST import AIMP python SDK
# import upper dir's python file
import sys
sys.path.append("../..") 
sys.path.append("..") 
import aimpInferWorkFlowSDK

#start init the aimpinferWorkFlowSDK
aimpPredict=aimpInferWorkFlowSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'yelp-polarity-triton'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
infer_host_FQDN=aimpPredict.infer_host_FQDN
infer_endpoint=aimpPredict.infer_endpoint
#end init the aimpinferSDK

tokenizer = BertTokenizer.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")
inputs = tokenizer(["Hello, my dog is cute"]) #pos
#inputs = tokenizer(["This movie is fantastic."]) #pos
#inputs = tokenizer(["This movie is fantastic,and my daughter likes it."]) #pos
#inputs = tokenizer(["Although this movie is fantastic,and my daughter likes it, I still hate it."]) #neg
#inputs = tokenizer(["Although this movie is fantastic,and my daughter likes it, I still hate it due to the terrible plots."]) #neg


data = {
   "inputs":[
   {
    "name": "input_0",
    "shape": [1,8],
    "datatype": "INT32",
    "data": inputs['input_ids']
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
r = requests.post(infer_endpoint, headers=headers, data=json.dumps(data), verify=False)
result = r.json()
pprint(result)

print('prediction probs:  ', result['outputs'][0]['data'])

labels = ['negtive','postive']
labels[np.array(result).argmax()]
