import base64
import json
import numpy as np 
import requests
import matplotlib.pyplot as plt

access_token="it-029f5933a3ff4bf88a1f5b10a0f26e31"
infer_host_FQDN="deeplabv3-resnet-101-torchserve.aimp.fat.aimpcloud.cn"
infer_endpoint="https://infer.fat.aimpcloud.cn/v1/models/deeplabv3-resnet-101-torchserve:predict"
image = open('./cat.jpg', 'rb') #open binary file in read mode
image_read = image.read()
image_64_encode = base64.b64encode(image_read)
bytes_array = image_64_encode.decode('utf-8')

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
print(result)
segmentation_result = np.array(result['predictions'])
predicted_class_3d = segmentation_result[..., 0].astype(int)
predicted_class = predicted_class_3d[0]  # 去掉第一维度
width, height = predicted_class.shape
# 定义颜色映射，每个类别对应一个颜色
class_colors = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128], 5: [128, 0, 128], 6: [0, 128, 128], 7: [128, 128, 128], 8: [64, 0, 0], 9: [192, 0, 0], 10: [64, 128, 0], 11: [192, 128, 0], 12: [64, 0, 128], 13: [192, 0, 128], 14: [64, 128, 128], 15: [192, 128, 128], 16: [0, 64, 0], 17: [128, 64, 0], 18: [0, 192, 0], 19: [128, 192, 0], 20: [0, 64, 128], 21: [128, 64, 128], 22: [0, 192, 128], 23: [128, 192, 128], 24: [64, 64, 0], 25: [192, 64, 0], 26: [64, 192, 0], 27: [192, 192, 0], 28: [0, 0, 64], 29: [128, 0, 64], 30: [0, 128, 64], 31: [128, 128, 64], 32: [0, 0, 192], 33: [128, 0, 192], 34: [0, 128, 192], 35: [128, 128, 192], 36: [64, 0, 64], 37: [192, 0, 64], 38: [64, 128, 64], 39: [192, 128, 64], 40: [64, 0, 192], 41: [192, 0, 192], 42: [64, 128, 192], 43: [192, 128, 192], 44: [0, 64, 64], 45: [128, 64, 64], 46: [0, 192, 64], 47: [128, 192, 64], 48: [0, 64, 192], 49: [128, 64, 192], 50: [64, 192, 192], 51: [192, 192, 192], 52: [32, 0, 0], 53: [160, 0, 0], 54: [32, 128, 0], 55: [160, 128, 0], 56: [32, 0, 128], 57: [160, 0, 128], 58: [32, 128, 128], 59: [160, 128, 128], 60: [96, 0, 0], 61: [224, 0, 0], 62: [96, 128, 0], 63: [224, 128, 0], 64: [96, 0, 128], 65: [224, 0, 128], 66: [96, 128, 128], 67: [224, 128, 128], 68: [32, 64, 0], 69: [160, 64, 0], 70: [32, 192, 0], 71: [160, 192, 0], 72: [32, 64, 128], 73: [160, 64, 128], 74: [32, 192, 128], 75: [160, 192, 128], 76: [96, 64, 0], 77: [224, 64, 0], 78: [96, 192, 0], 79: [224, 192, 0], 80: [64, 64, 128]}
# 创建空白图像，与输入图像相同大小
output_image = np.zeros((width, height, 3), dtype=np.uint8)
# 遍历每个像素，根据类别为每个像素着色
for row in range(width):
    for col in range(height):
        class_idx = int(predicted_class[row, col])
        if class_idx in class_colors:
            output_image[row, col, :] = class_colors[class_idx]

# 显示分割图
plt.imshow(output_image)
plt.axis('off')
plt.show()