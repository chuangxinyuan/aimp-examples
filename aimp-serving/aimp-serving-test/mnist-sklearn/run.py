import json
import numpy as np 
import requests
from pprint import pprint
import matplotlib.pyplot as plt
infer_endpoint = "https://infer.fat.aimpcloud.cn/v1/models/mnist2:predict"
data = {
    "instances": [
      [ 0,  0,  2, 13, 16,  8,  0,  0,  0,  0, 11, 16,  6,
        2,  0,  0,  0,  2, 16,  8,  0,  0,  0,  0,  0,  5,
       16,  9,  1,  0,  0,  0,  0,  5, 16, 16, 13,  2,  0,
        0,  0,  1, 16,  6,  8, 14,  0,  0,  0,  0, 11, 10,
        1, 16,  5,  0,  0,  0,  3, 15, 16, 16,  3,  0]
    ]
}
sample_image = np.array(data["instances"])[0].reshape(8, 8)  # Reshape to the original image dimensions
plt.imshow(sample_image, cmap='gray')
plt.show()
json_str = json.dumps(data)

# 将JSON字符串编码为字节串
data = json_str.encode('utf-8')
# 
headers = {
    'infer-access-token': "it-029f5933a3ff4bf88a1f5b10a0f26e31",
    'Content-Type': 'application/json',
    'Host': "mnist2.aimp.fat.aimpcloud.cn",
}
r = requests.post(infer_endpoint, headers=headers, data=data, verify=False)
print(r)
result = r.json()

pprint(result)

