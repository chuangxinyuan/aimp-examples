# 介绍

### 例子：手写体数据集

手写体数据集是sklearn自带的数据集，特征是1797*64的二维数组，代表数据集中1797个样本，每一个样本图片均是一维向量，特征维度为64，每个像素点代表1维的特征，其取值范围为0-16；标签共有10个类别，其取值范围为0-9中的任意一个数字。

### 算法介绍

1. 算法名称：RF(随机森林)
2. 输入：1797*64的二维数组，代表数据集中1797个样本，每一个样本图片均是一维向量，特征维度为64，每个像素点代表1维的特征，其取值范围为0-16；
3. 输出：0-9中的任意一个数字

### 中台部署流程

1. 运行random-forest-train.py，将生成model.joblib文件
   本地joblib的版本需为1.0.1

2. 将model.joblib打包成zip文件，model.zip
   关系如下：
   |model.zip
   |—model.joblib

3. 创建random-forest.yaml文件

4. 运行run.py

   在run.py中，需要**更新infer_endpoint、infer-access-token、Host**三个变量，其需要在中台中获取，对应关系如下：
   infer_endpoint 为访问地址；infer-access-token为访问令牌；Host为目标服务。

   run.py运行成功将输出一张手写体图片和对应的标签

   ![image-20231009101458083](C:\Users\liuyinglai\AppData\Roaming\Typora\typora-user-images\image-20231009101458083.png)

   ```python
   <Response [200]> {'predictions': [8]}
   ```

   

### 参考

- [参考 sikit learn 官方安装文档](https://scikit-learn.org/stable/install.html)
- [参考 kferving sklearn官方文档](https://github.com/chuangxinyuan/aimp-kfserving/tree/release-0.6/docs/samples/v1beta1/sklearn/v1)