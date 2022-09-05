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

#start init the aimpinferWorkFlowSDK
aimpPredict=aimpInferWorkFlowSDK.aimpInfer()
aimpPredict.namespace = 'mp'
aimpPredict.model_name = 'yelp-polarity-triton'
aimpPredict.getAccess()
access_token=aimpPredict.api_access_token
endpoint=aimpPredict.infer_endpoint
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
    'onepanel-access-token': access_token
}

r = requests.post(endpoint,headers=headers, json=data)

result = r.json()

print('prediction probs:  ', result['outputs'][0]['data'])

labels = ['negtive','postive']
labels[np.array(result).argmax()]
