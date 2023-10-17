import json
import numpy as np 
import requests
from pprint import pprint
import matplotlib.pyplot as plt
infer_endpoint = "https://infer.fat.aimpcloud.cn/v1/models/gbdt:predict"
data = {
    "instances": [
      [ 0,  0,  0,  2, 13,  0,  0,  0,  0,  0,  0,  8, 15,
        0,  0,  0,  0,  0,  5, 16,  5,  2,  0,  0,  0,  0,
       15, 12,  1, 16,  4,  0,  0,  4, 16,  2,  9, 16,  8,
        0,  0,  0, 10, 14, 16, 16,  4,  0,  0,  0,  0,  0,
       13,  8,  0,  0,  0,  0,  0,  0, 13,  6,  0,  0]

    ]
}
sample_image = np.array(data["instances"])[0].reshape(8, 8)  # Reshape to the original image dimensions
plt.imshow(sample_image, cmap='gray')
plt.show()

json_str = json.dumps(data)

# 将JSON字符串编码为字节串
data = json_str.encode('utf-8')
headers = {
    'infer-access-token': "it-029f5933a3ff4bf88a1f5b10a0f26e31",
    'Content-Type': 'application/json',
    'Host': "gbdt.aimp.fat.aimpcloud.cn",
}
print('---Prediction RESULTS---')
r = requests.post(infer_endpoint, headers=headers, data=data, verify=False)
print(r)
result = r.json()

pprint(result)

