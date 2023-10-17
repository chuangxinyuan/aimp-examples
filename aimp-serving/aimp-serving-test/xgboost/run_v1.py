import json
import numpy as np 
import requests
from pprint import pprint
infer_endpoint = "https://infer.fat.aimpcloud.cn/v1/models/xgboost:predict"

data = {
    "instances": [
      [6.8,  2.8,  4.8,  1.4],
      [6.0,  3.4,  4.5,  1.6]
    ]
}

json_str = json.dumps(data)

data = json_str.encode('utf-8')
# 
headers = {
    'infer-access-token': "it-029f5933a3ff4bf88a1f5b10a0f26e31",
    'Content-Type': 'application/json',
    'Host': "xgboost.aimp.fat.aimpcloud.cn",
}
print('---Prediction RESULTS---')
r = requests.post(infer_endpoint, headers=headers, data=data, verify=False)
print(r)
result = r.json()
print ("预测结果说明： 0->山鸢尾（setosa）、1->变色鸢尾（versicolor）、2->维吉尼亚鸢尾（virginica）")
pprint(result)

