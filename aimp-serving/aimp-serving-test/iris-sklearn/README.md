
# 介绍
## 例子：鸢尾花（Iris）多分类
iris的分类是一个典型的人工智能分类问题，选取的是比较典型特点的三种鸢尾花：山鸢尾Iris setosa(0)、变色鸢尾Iris versicolor (1)、维吉尼亚鸢尾Iris virginica (2)，通过花的四个特征确定了单株鸢尾花的种类，这个四个特征是鸢尾花花瓣（petals）的长度和宽度、花萼（sepals）的长度和宽度，关于该例子的其他详细资料请参考：[ 鸢尾花（Iris）](https://blog.csdn.net/heivy/article/details/100512264)

本例子模型通过输入4个特征值，最终推测花的种类，结果值分别是0, 1, 2，
## 算法介绍
1. 算法名称：SVM （支持向量机）
2. 输入：为N*4的矩阵，每一行表示一个待预测的花的特征，特征值有4个，是鸢尾花花瓣（petals）的长度和宽度、花萼（sepals）的长度和宽度
3. 输出：为0，1，2，分别对应着如下三种花的种类，山鸢尾（setosa）、变色鸢尾（versicolor）、维吉尼亚鸢尾（virginica）这三个名词都是花的品种。

## Creating your own model and testing the SKLearn Server locally.
To test the [Scikit-Learn](https://scikit-learn.org/stable/) server, first we need to generate a simple scikit-learn model using Python. 
# 创建你的模型
sklearn 模型文件格式是 joblib， 参考如下的例子创建自己的模型

* 创建模型示例如下：

```python 3.7
from sklearn import svm
from sklearn import datasets
from joblib import dump
clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
dump(clf, 'model.joblib')
```
* 注意，本地测试的时候，本地安装的Scikit-learn的版本最好和AIMP的OOB的模型服务器中的sikit-learn的版本一致，即0.20.3版本，这样能够保证本地测试的效果和在kfserving中运行的效果保持一致。

# 参考
* [参考 sikit learn 官方安装文档](https://scikit-learn.org/stable/install.html)
* [参考 kferving sklearn官方文档](https://github.com/chuangxinyuan/aimp-kfserving/tree/release-0.6/docs/samples/v1beta1/sklearn/v1)


