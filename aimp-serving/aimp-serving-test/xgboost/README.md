# 介绍

### 例子：鸢尾花（Iris）多分类

iris的分类是一个典型的人工智能分类问题，选取的是比较典型特点的三种鸢尾花：山鸢尾Iris setosa(0)、变色鸢尾Iris versicolor (1)、维吉尼亚鸢尾Iris virginica (2)，通过花的四个特征确定了单株鸢尾花的种类，这个四个特征是鸢尾花花瓣（petals）的长度和宽度、花萼（sepals）的长度和宽度，关于该例子的其他详细资料请参考： 鸢尾花（Iris）

本例子模型通过输入4个特征值，最终推测花的种类，结果值分别是0, 1, 2，

### 算法介绍

https://xgboost.readthedocs.io/en/stable/

## 中台部署模型流程

1. 运行xgboost-train.py，将生成model.bst文件；

2. 将model.bst文件生成zip压缩包，文件名为model.zip.
   文件目录如下：
   |model.zip
   |—model.bst

3. 创建xgboost.yaml文件

4. 运行run.py

   在run.py中，需要**更新infer_endpoint、infer-access-token、Host**三个变量，其需要在中台中获取，对应关系如下：
   infer_endpoint 为访问地址；infer-access-token为访问令牌；Host为目标服务。

   - **需要注意的是**
     如果xgboost.yaml中protocolVersion: "v2"，那么就运行run-v2.py

     如果xgboost.yaml中protocolVersion: "v1"，那么就运行run-v1.py

     **v2和v1的infer_endpoint不同**

     - 在v2中是/v2/models/xgboost/infer
     - 在v1中是 /v1/models/xgboost:predict

     **v2和v1的输入形式不同**

     - v2

     ```python
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
     ```

     运行成功将输入如下内容

     ```python
     <Response [200]>
     {'id': '5212ac81-a097-448a-b55d-8ca33dcca2e4',
      'model_name': 'xgboost',
      'model_version': None,
      'outputs': [{'data': [1.0, 1.0],
                   'datatype': 'FP32',
                   'name': 'predict',
                   'parameters': None,
                   'shape': [2]}],
      'parameters': None}
     ```
     
     - v1
     
       ```python
       data = {
           "instances": [
             [6.8,  2.8,  4.8,  1.4],
             [6.0,  3.4,  4.5,  1.6]
           ]
       }
       ```

建议使用V2版本

### v2和v1具体内容参考

- [aimp-kfserving/docs/README.md at release-0.6 · chuangxinyuan/aimp-kfserving (github.com)](https://github.com/chuangxinyuan/aimp-kfserving/blob/release-0.6/docs/README.md)
- [参考 sikit learn 官方安装文档](https://scikit-learn.org/stable/install.html)
- [参考 kferving sklearn官方文档](https://github.com/chuangxinyuan/aimp-kfserving/tree/release-0.6/docs/samples/v1beta1/sklearn/v1)

