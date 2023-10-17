from sklearn.ensemble import GradientBoostingClassifier  # Import GBDT classifier
from sklearn import datasets
from joblib import dump
from pprint import pprint

digits = datasets.load_digits()
X, y = digits.data, digits.target
clf = GradientBoostingClassifier(n_estimators=100, random_state=0)  # Adjust parameters as needed
clf.fit(X, y)
dump(clf, 'model.joblib')
