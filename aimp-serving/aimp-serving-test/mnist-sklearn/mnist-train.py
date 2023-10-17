from sklearn import svm
from sklearn import datasets
from joblib import dump
from pprint import pprint
# 加载鸢尾花数据集
digits = datasets.load_digits()
X, y = digits.data, digits.target
clf = svm.SVC(gamma='scale')
# 训练模型
clf.fit(X, y)
# 保存模型到文件
dump(clf, 'modeljoblib')

