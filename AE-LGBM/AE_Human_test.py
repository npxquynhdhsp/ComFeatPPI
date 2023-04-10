"""
# # TÁI LẬP TRÌNH TỪ BÀI BÁO GỐC
# Sử dụng siêu tham số từ bài báo gốc
# Sử dụng chương trình trích xuất đặc trưng được cung cấp từ bài báo gốc
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model

# %%

# os.chdir('_original/Human')
os.chdir('_original/Yeast')
print(os.getcwd())

label = pd.read_csv("label.csv")
# print(label)
feat_a = pd.read_csv("total_features_a.csv")
# print(feat_a)
feat_b = pd.read_csv("total_features_b.csv")
# print(feat_b)
feat_all = pd.concat([feat_a, feat_b], axis=1)
print(feat_all.shape)

X_no_enc = feat_all.values
y_no_enc = label.values


# %%


def AE_net(n_col, k=208):
    model_AE = Sequential(
        [
            Dense(k, activation='sigmoid', input_shape=(n_col,)),
            Dropout(rate=0.2),
            Dense(n_col, activation='sigmoid', input_shape=(k,))
        ]
    )

    # Constructing Autoencoder
    model_AE.compile(optimizer='nadam',
                     loss='mean_squared_error',
                     metrics="accuracy")

    # model_AE1.summary()
    enc_net = Model(inputs=model_AE.input,
                    outputs=model_AE.layers[1].output)
    # enc_net.summary()
    return model_AE, enc_net


# %%

k = 208
n_col = feat_a.shape[1]
model_AE1, enc_a = AE_net(n_col, k)

# Fitting the Autoencoder
print("\nHuan luyen AE 1...")
print("feat_a", feat_a.shape)
history_a = model_AE1.fit(feat_a, feat_a,
                          epochs=200,
                          batch_size=50,
                          validation_split=0.2,
                          verbose=0)

feat_a_enc = enc_a.predict(feat_a)
print("feat_a_enc", feat_a_enc.shape)

# %%

print("\nHuan luyen AE 2...")
print("feat_b", feat_b.shape)
model_AE2, enc_b = AE_net(n_col, k)
history_b = model_AE2.fit(feat_b, feat_b,
                          epochs=200,
                          batch_size=50,
                          validation_split=0.2,
                          verbose=0)

feat_b_enc = enc_b.predict(feat_b)
print(feat_b_enc.shape)

# %%

gbm = LGBMClassifier(n_estimators=500,  # @200 hoặc 250
                     num_leaves=80,
                     learning_rate=0.1,  # @0.1
                     random_seed=0
                     )

feat_all_enc = np.concatenate((feat_a_enc, feat_b_enc), axis=1)
print('\n--- Human new trainning set', feat_all_enc.shape)
X_tr = np.copy(feat_all_enc)
y_tr = np.copy(label).flatten()
y_tr[y_tr == -1] = 0
print(y_tr)

gbm.fit(X_tr, y_tr)


# %%

def eval_testset(dset_name):
    os.chdir(r'D:\NCSI\3 - Thuc nghiem\ModifiedModel\code_PPI_new_OK\AE_LGBM_2020, full paper\AE_LGBM_2020\mod_AE_LGBM\Independent Species/' + dset_name)
    feat_a = pd.read_csv("total_features_a.csv")
    feat_a = enc_a.predict(feat_a)

    feat_b = pd.read_csv("total_features_b.csv")
    feat_b = enc_b.predict(feat_b)

    te_X = np.concatenate((feat_a, feat_b), axis=1)
    print('\n', dset_name, ', n_samples', len(te_X), end=', ')
    te_y = [1] * te_X.shape[0]

    prob_y = gbm.predict_proba(te_X)

    pickle.dump(prob_y, open(r''
                             r'D:\NCSI\3 - Thuc nghiem\ModifiedModel\code_PPI_new_OK\AE_LGBM_2020, full paper\AE_LGBM_2020\mod_AE_LGBM\Independent Species\__LUU_teston_Yeastcore/' + dset_name + '_pred.pkl', 'wb'))
    pred_y = gbm.predict(te_X).flatten()
    # print(pred_y)
    print(dset_name, "acc {:.4f}".format(sum(pred_y == te_y) / len(te_y)))


# eval_testset('Celeg')
# eval_testset('Ecoli')
# eval_testset('Hsapi')
eval_testset('Hpylo')
# eval_testset('Mmusc')
# eval_testset('Dmela')
#
# eval_testset('Wnt')
# eval_testset('CD9')
# eval_testset('Cancer_specific')
# eval_testset('ras-ref-erk_network')

# %%
#
# Ecoli, n_samples
# 6954, acc
# 0.9971
# Hsapi, n_samples
# 1412, acc
# 0.9901
# Mmusc, n_samples
# 313, acc
# 0.9968
# Dmela, n_samples
# 21975, acc
# 0.9966
# Wnt, n_samples
# 96, acc
# 0.9896
# CD9, n_samples
# 16, acc
# 1.0000
# Cancer_specific, n_samples
# 108, acc
# 0.9815
# ras - ref - erk_network, n_samples
# 198, acc
# 0.9949


# Celeg , n_samples 4013, acc 0.9983
#  Ecoli , n_samples 6954, acc 0.9971
#  Hsapi , n_samples 1412, acc 0.9901
#  Mmusc , n_samples 313, acc 0.9968
#  Dmela , n_samples 21975, acc 0.9966
#  Wnt , n_samples 96, acc 0.9896
#  CD9 , n_samples 16, acc 1.0000
#  Cancer_specific , n_samples 108, acc 0.9815
#  ras-ref-erk_network , n_samples 198, acc 0.9949
# %%
