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
import onepanel.core.api
from onepanel.core.api.rest import ApiException
import onepanel.core.auth
from pprint import pprint


# ## Get Onepanel Access Token for network requests

# In[2]:


# If inside of Onepanel, get mounted service account token to use as API Key
access_token = onepanel.core.auth.get_access_token()

print('---ONEPANEL_API_URL----', os.getenv('ONEPANEL_API_URL'))
# Configure API key authorization: Bearer
configuration = onepanel.core.api.Configuration(
    host=os.getenv('ONEPANEL_API_URL'),
    api_key={
        'authorization': access_token
    }
)
configuration.api_key_prefix['authorization'] = 'Bearer'


# In[5]:


namespace = 'mp'
model_name = 'yolox-onnx-triton'


# In[6]:


# Get status, endpoint
with onepanel.core.api.ApiClient(configuration) as api_client:
    api_instance = onepanel.core.api.InferenceServiceApi(api_client)

    try:
        ready = False
        while not ready:
            api_response = api_instance.get_inference_service(namespace, model_name)
            ready = api_response.ready
            endpoint = api_response.predict_url
            print('---api_response.predict_url---', endpoint)
            time.sleep(1)
    except ApiException as e:
        print("Exception when calling InferenceServiceApi->get_inference_service_status: %s\n" % e)


# In[8]:

with open('./img.pkl','rb') as f:
    # shape of img_data: [1,3, 224, 224]
    img_data = pickle.load(f).astype('float32').tolist()

data = {
   "inputs":[
   {
    "name": "input",
    "shape": [1,3,224,224],
    "datatype": "FP32",
    "data": img_data
   }
   ]
}



headers = {
    'onepanel-access-token': access_token
}


r = requests.post(endpoint, headers=headers, json=data)

result = r.json()

print(result)

#result 说明
    
#labels: cls_label
#dets: xyxy boxes with the prob (flattened)
'''
{'model_name': 'yolox-onnx-triton', 'model_version': '1', 'outputs': 
[{'name': 'dets', 'datatype': 'FP32', 'shape': [1, 3, 5], 'data': [126.69447326660156, 76.2667465209961, 180.7909698486328, 177.94143676757812, 0.887129545211792, 60.801849365234375, 24.55431365966797, 106.42037963867188, 202.824951171875, 0.8643292188644409, 0.0, 0.0, 0.0, 0.0, 0.0]},
 {'name': 'labels', 'datatype': 'INT64', 'shape': [1, 3], 'data': [0, 0, 0]}]}
'''
# cls_label 到 cls_name 可以从下面的coco128_names进行查询
coco128_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names
