import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

# os.chdir('C:/Protein')
rootdir = sys.path[1]
print(rootdir)

# load or create your dataset
os.chdir(rootdir + r'\\' + "Human")
Mydata = pd.read_csv("gbm_Human.csv", header=0)
Mydata.columns.name = None

data = Mydata.drop(Mydata.columns[Mydata.shape[1] - 1], axis=1)
label = Mydata.iloc[:, (Mydata.shape[1] - 1)]

data_tr = data
print("data_tr", data_tr.shape)
label_tr = label

##########################################################################
# thnhan's
# load the model from file
encoder_a = load_model('AE_Human_a_.h5')
encoder_a = Model(inputs=encoder_a.input, outputs=encoder_a.layers[1].output)
encoder_b = load_model('AE_Human_b_.h5')
encoder_b = Model(inputs=encoder_b.input, outputs=encoder_b.layers[1].output)

# load test
os.chdir(r'/Independent Species/Ecoli')
feat_a = pd.read_csv("total_features_a_new.csv")

print(feat_a)
feat_a = encoder_a.predict(feat_a)
feat_b = pd.read_csv("total_features_b_new.csv")
feat_b = encoder_b.predict(feat_b)
data_te = np.concatenate((feat_a, feat_b), axis=1)
print('data_te', data_te.shape)
label_te = [1] * data_te.shape[0]
##########################################################################

# ##########################################################################
# # Author's
# os.chdir(r'/Independent Species/authors')
# test = pd.read_csv("gbm_ecoli.csv", header=0)
# test.columns.name = None
# data_te = test.drop(test.columns[test.shape[1] - 1], axis=1)
# label_te = test.iloc[:, (test.shape[1] - 1)]
# ##########################################################################

# # create dataset for lightgbm
lgb_train = lgb.Dataset(data_tr, label_tr)
# lgb_eval = lgb.Dataset(data_te, label_te, reference=lgb_train)
# # specify your configurations as a dict
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'num_leaves': 80,  # 50 for Yeast
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }
#
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=2000,
#                 # valid_sets=lgb_eval,
#                 # early_stopping_rounds=100,
#                 # verbose_eval=0
#                 )

gbm = lgb.LGBMClassifier(
    n_estimators=500,
    # num_boost_round=2000,
    num_leaves=80,
    learning_rate=0.05,
).fit(data_tr, label_tr)

# gbm.save_model(["model"+str(k)+".txt"][0])
print("Huan luyen xong")
print("\nTest...")

y_pred = gbm.predict(data_te,
                     # num_iteration=gbm.best_iteration
                     )

for i in range(0, len(list(y_pred))):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = -1

# label_te.reset_index(drop=True, inplace=True)
# pd.DataFrame(list(y_pred)).reset_index(drop=True, inplace=True)

print(f"Điểm test  {accuracy_score(label_te, y_pred):0.4f}")
print(f"Điểm train {gbm.score(data_tr, label_tr):0.4f}")

# # Confusion matrix
# cm = confusion_matrix(label_te, y_pred)
# pd.DataFrame(cm)
# Tp = cm[1, 1]
# Fp = cm[0, 1]
# Tn = cm[0, 0]
# Fn = cm[1, 0]
# Sn = Tp / (Tp + Fn)
# Sp = Tn / (Tn + Fp)
# Acc = (Tp + Tn) / np.sum(cm)
# Mcc = (Tp * Tn - Fp * Fn) / ((((Tp + Fp) * (Tp + Fn)) ** (1 / 2)) * (((Tn + Fp) * (Tn + Fn)) ** (1 / 2)))
