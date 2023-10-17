import json
import numpy as np 
import requests
from pprint import pprint
import matplotlib.pyplot as plt
infer_endpoint = "https://infer.fat.aimpcloud.cn/v1/models/rf:predict"
data = {
    "instances": [[ 0,  0,  3, 10, 14,  3,  0,  0,  0,  8, 16, 11, 10,
       13,  0,  0,  0,  7, 14,  0,  1, 15,  2,  0,  0,  2,
       16,  9, 16, 16,  1,  0,  0,  0, 12, 16, 15, 15,  2,
        0,  0,  0, 12, 10,  0,  8,  8,  0,  0,  0,  9, 12,
        4,  7, 12,  0,  0,  0,  2, 11, 16, 16,  9,  0]
    ]
}
sample_image = np.array(data["instances"])[0].reshape(8, 8)  # Reshape to the original image dimensions
plt.imshow(sample_image, cmap='gray')
plt.show()
json_str = json.dumps(data)

# 将JSON字符串编码为字节串
data = json_str.encode('utf-8')
# print(type())
# f = open('./iris-input.json', 'rb') #open binary file in read mode
# data = f.read()
# print(type(data))
headers = {
    'infer-access-token': "it-029f5933a3ff4bf88a1f5b10a0f26e31",
    'Content-Type': 'application/json',
    'Host': "rf.aimp.fat.aimpcloud.cn",
}
# print('---api_predict_endpoint and headers---')
# # print (infer_endpoint)
# # pprint(headers)
# # print('\n')
print('---Prediction RESULTS---')
r = requests.post(infer_endpoint, headers=headers, data=data, verify=False)
print(r)
result = r.json()

pprint(result)

