"""
Taken from:
@inproceedings{upadhyay2021case,
  title={A Case-Based Approach to Data-to-Text Generation},
  author={Upadhyay, Ashish and Massie, Stewart and Singh, Ritwik Kumar and Gupta, Garima and Ojha, Muneendra},
  booktitle={International Conference on Case-Based Reasoning},
  pages={232--247},
  year={2021},
  organization={Springer}
}
"""

import pickle, argparse
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount
from tpot.export_utils import set_param_recursive
from sklearn.metrics import accuracy_score, f1_score

class ImpPlayerClassifier:

    def train_clf(self, features, target):
        exported_pipeline = make_pipeline(
            ZeroCount(),
            MLPClassifier(alpha=0.0001, learning_rate_init=0.01)
        )
        set_param_recursive(exported_pipeline.steps, 'random_state', 42)
        exported_pipeline.fit(features, target)
        return exported_pipeline

    def prediction(self, model, features):
        return model.predict(features)
    
    def score(self, predictions, target):
        return {
            'accuracy': accuracy_score(target, predictions),
            'f1': f1_score(target, predictions, average='macro')
        }

argparser = argparse.ArgumentParser()
argparser.add_argument('-season', '--season', type=str, default='2014', \
                        choices=['2014', '2015', '2016', '2017', '2018', 'all', 'bens', 'juans'])
args = argparser.parse_args()
season = args.season
print(args, season)

data_dir = f'player_clf/data/{season}'

train_x = np.load(f'{data_dir}/X_train.npz')['arr_0']
train_y = np.load(f'{data_dir}/y_train.npz')['arr_0']
val_x = np.load(f'{data_dir}/X_validation.npz')['arr_0']
val_y = np.load(f'{data_dir}/y_validation.npz')['arr_0']
test_x = np.load(f'{data_dir}/X_test.npz')['arr_0']
test_y = np.load(f'{data_dir}/y_test.npz')['arr_0']
X_train = np.concatenate((train_x, val_x))
y_train = np.concatenate((train_y, val_y))
print(X_train.shape, y_train.shape, test_x.shape, test_y.shape)

ipc_obj = ImpPlayerClassifier()
model = ipc_obj.train_clf(X_train, y_train)
pickle.dump(model, open(f'player_clf/model/model_{season}.pkl', 'wb'))
model = pickle.load(open(f'player_clf/model/model_{season}.pkl', 'rb'))
pred_y = ipc_obj.prediction(model, test_x)
print(ipc_obj.score(pred_y, test_y))
