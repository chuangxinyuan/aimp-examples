from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from joblib import dump
from joblib import dump
from pprint import pprint

digits = datasets.load_digits()
X, y = digits.data, digits.target
sample_image = X[500].reshape(8, 8)
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
dump(clf, 'model.joblib')
