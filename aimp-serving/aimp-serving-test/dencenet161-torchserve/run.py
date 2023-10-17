import base64
import json
import time
import numpy as np 
import requests
from PIL import Image

access_token="it-029f5933a3ff4bf88a1f5b10a0f26e31"
infer_host_FQDN="densenet161-torchserve.aimp.fat.aimpcloud.cn"
infer_endpoint="https://infer.fat.aimpcloud.cn/v1/models/densenet161-torchserve:predict"

image = open('./cat.jpg', 'rb') #open binary file in read mode
image_read = image.read()
image_64_encode = base64.b64encode(image_read)
bytes_array = image_64_encode.decode('utf-8')

image_show = Image.open('./cat.jpg')
image_show.show()

data = {
    'instances': [
        {'data': bytes_array}
    ]
}
headers = {
    
    'infer-access-token': access_token,
    'Content-Type': 'application/json',
    'Host': infer_host_FQDN,
}

print('---Prediction RESULTS---')

r = requests.post(infer_endpoint, headers=headers, data=json.dumps(data), verify=False)
result = r.json()
    
print("predict label:", list(result['predictions'][0].keys())[0])