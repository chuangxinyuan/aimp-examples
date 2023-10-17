import lightgbm as lgb
from sklearn.datasets import load_iris
import os

model_dir = "."
BST_FILE = "model1.bst"

iris = load_iris()
y = iris['target']
X = iris['data']
dtrain = lgb.Dataset(X, label=y)

params = {
    'objective':'multiclass', 
    'metric':'softmax',
    'num_class': 3
}
lgb_model = lgb.train(params=params, train_set=dtrain)
model_file = os.path.join(model_dir, BST_FILE)
lgb_model.save_model(model_file)