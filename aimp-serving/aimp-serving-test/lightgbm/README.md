# 介绍

### 例子：鸢尾花（Iris）多分类

iris的分类是一个典型的人工智能分类问题，选取的是比较典型特点的三种鸢尾花：山鸢尾Iris setosa(0)、变色鸢尾Iris versicolor (1)、维吉尼亚鸢尾Iris virginica (2)，通过花的四个特征确定了单株鸢尾花的种类，这个四个特征是鸢尾花花瓣（petals）的长度和宽度、花萼（sepals）的长度和宽度，关于该例子的其他详细资料请参考： 鸢尾花（Iris）

本例子模型通过输入4个特征值，最终推测花的种类，结果值分别是0, 1, 2，

### 算法介绍

1. 算法名称：lightGBM
2. 算法详解：https://lightgbm.readthedocs.io/en/latest/index.html

## 中台部署模型流程

1. 运行lightgbm-train.py，将生成model.bst文件；

2. 将model.bst文件生成zip压缩包，文件名为model.zip.
   文件目录如下：
   |model.zip
   |—model.bst

3. 创建lightgbm.yaml文件
   需要说明的是，yaml文件中不要添加runtimeVersion字段

4. 运行run.py

   在run.py中，需要**更新infer_endpoint、infer-access-token、Host**三个变量，其需要在中台中获取，对应关系如下：
   infer_endpoint 为访问地址；infer-access-token为访问令牌；Host为目标服务。
数据的输入格式严格按照run.py或inputs.json中的格式

### 参考

- [参考 sikit learn 官方安装文档](https://scikit-learn.org/stable/install.html)
- [参考 kferving sklearn官方文档](https://github.com/chuangxinyuan/aimp-kfserving/tree/release-0.6/docs/samples/v1beta1/sklearn/v1)

### 注意

yaml文件中lightgbm的runtimeversion 不能写，查看后台发现会拉取lgbserver：latest镜像

[kfserving/lgbserver - Docker Image | Docker Hub](https://hub.docker.com/r/kfserving/lgbserver/tags)

建议拉取v0.6.1版本的镜像，保存在cxy的镜像容器中

docker pull kfserving/lgbserver:v0.6.1
