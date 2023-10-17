import json
import numpy as np 
import requests
from pprint import pprint
infer_endpoint = "https://infer.fat.aimpcloud.cn/v2/models/xgboost/infer"
data = {
  "inputs": [
    {
      "name": "input-0",
      "shape": [2, 4],
      "datatype": "FP32",
      "data": [
        [6.8, 2.8, 4.8, 1.4],
        [6.0, 3.4, 4.5, 1.6]
      ]
    }
  ]
}

json_str = json.dumps(data)

# 将JSON字符串编码为字节串
data = json_str.encode('utf-8')
# 
headers = {
    'infer-access-token': "it-029f5933a3ff4bf88a1f5b10a0f26e31",
    'Content-Type': 'application/json',
    'Host': "xgboost.aimp.fat.aimpcloud.cn",
}
print('---api_predict_endpoint and headers---')
print('---Prediction RESULTS---')
r = requests.post(infer_endpoint, headers=headers, data=data, verify=False)
print(r)
result = r.json()
print ("预测结果说明： 0->山鸢尾（setosa）、1->变色鸢尾（versicolor）、2->维吉尼亚鸢尾（virginica）")
pprint(result)

