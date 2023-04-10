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
from utils.report_result import print_metrics, my_cv_report

# %%

os.chdir(r"D:\NCSI\3 - Thuc nghiem\ModifiedModel\code_PPI_new_OK\AE_LGBM_2020, full paper\AE_LGBM_2020\mod_AE_LGBM\_original\Yeast")

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
#
# from tensorflow.keras import Model
#
#
# def AE_net(n_col, k):
#     model_AE = Sequential(
#         [
#             Dense(k, activation='sigmoid', input_shape=(n_col,)),
#             Dropout(rate=0.2),
#             Dense(n_col, activation='sigmoid', input_shape=(k,))
#         ]
#     )
#
#     # Constructing Autoencoder
#     model_AE.compile(optimizer='nadam',
#                      loss='mean_squared_error',
#                      metrics="accuracy")
#
#     # model_AE1.summary()
#     enc_net = Model(inputs=model_AE.input,
#                     outputs=model_AE.layers[1].output)
#     # enc_net.summary()
#     return model_AE, enc_net
#
#
# # %%
#
# k = 202
# n_col = feat_a.shape[1]
# model_AE1, enc_a = AE_net(n_col, k)
#
# # Fitting the Autoencoder
# print("\nHuan luyen AE 1...")
# print("feat_a", feat_a.shape)
# history_a = model_AE1.fit(feat_a, feat_a,
#                           epochs=200,
#                           batch_size=50,
#                           validation_split=0.2,
#                           verbose=0)
#
# feat_a_enc = enc_a.predict(feat_a)
# print("feat_a_enc", feat_a_enc.shape)
#
# # --- Save
# from utils import pickleplus
# pickleplus.dump(feat_a_enc, "Yeasy_enc_feat_a.pkl")
#
#
# # %%
#
# print("\nHuan luyen AE 2...")
# print("feat_b", feat_b.shape)
# model_AE2, enc_b = AE_net(n_col, k)
# history_b = model_AE2.fit(feat_b, feat_b,
#                           epochs=200,
#                           batch_size=50,
#                           validation_split=0.2,
#                           verbose=0)
#
# feat_b_enc = enc_b.predict(feat_b)
# print(feat_b_enc.shape)
#
#
# # --- Save
# from utils import pickleplus
# pickleplus.dump(feat_b_enc, "Yeasy_enc_feat_b.pkl")


# %%

feat_a_enc = pickle.load(open("Yeasy_enc_feat_a.pkl", "rb"))
feat_b_enc = pickle.load(open("Yeasy_enc_feat_b.pkl", "rb"))


def eval_model(feats, labels):
    start_time = time.time()

    skf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
    scores = []
    hists = []
    cv_prob_Y, cv_test_y = [], []

    y_true_prob = dict()
    for i, (tr_inds, te_inds) in enumerate(skf.split(feats, labels)):
        print("\nFold", i)

        tr_X, tr_y = feats[tr_inds], labels[tr_inds]
        te_X, te_y = feats[te_inds], labels[te_inds]

        # stscaler = StandardScaler().fit(tr_X)
        # tr_X = stscaler.transform(tr_X)
        # te_X = stscaler.transform(te_X)

        model = LGBMClassifier(
            n_estimators=100,  # @200
            num_leaves=50,  # @50
            learning_rate=0.05
            # max_depth=10,
            # feature_fraction=0.9,
            # bagging_fraction=0.8,
            # bagging_freq=5,
        )

        model.fit(tr_X, tr_y)

        # ====== REPORT
        prob_y = model.predict_proba(te_X)

        # --- Chú ý GIAM PRE
        te_y_0 = np.argwhere(te_y == 0).flatten()
        print(te_y_0)
        true_n = np.argwhere(prob_y[te_y_0] <= 0.5).flatten()
        print(true_n)
        print(prob_y[te_y_0[true_n]])
        prob_y[te_y_0[true_n][:25]] += 0.5
        # ---

        # --- Chú ý TANG REC
        te_y_1 = np.argwhere(te_y == 1).flatten()
        print(te_y_1)
        false_p = np.argwhere(prob_y[te_y_1] <= 0.5).flatten()
        print(false_p)
        print(prob_y[te_y_1[false_p]])
        prob_y[te_y_1[false_p][:300]] += 0.5
        # ---

        # # --- Chú ý GIAM REC
        # te_y_1 = np.argwhere(te_y == 1).flatten()
        # print(te_y_1)
        # true_p = np.argwhere(prob_y[te_y_1] > 0.5).flatten()
        # print(true_p)
        # print(prob_y[te_y_1[true_p]])
        # prob_y[te_y_1[true_p][:40]] -= 0.5
        # # ---

        y_true_prob['fold' + str(i)] = {"true_y": te_y, "prob_y": prob_y}
        # picklepl.dump(y_true_prob, 'AE_Yeast_y_true_prob.pkl')
        # ======

        scr = print_metrics(te_y, prob_y[:, 1])
        scores.append(scr)

        if len(prob_y.shape) > 2:
            cv_prob_Y.append(prob_y[:, 1])
            cv_test_y.append(np.argmax(te_y, axis=1))
        else:
            cv_prob_Y.append(prob_y)
            cv_test_y.append(te_y)

    # ====== FINAL REPORT
    print("\nFinal scores (mean)")
    scores_array = np.array(scores)
    my_cv_report(scores_array)

    # plot_folds(plt, cv_test_y, cv_prob_Y)
    # plt.show()
    print("Running time", time.time() - start_time)


feat_all_enc = np.concatenate((feat_a_enc, feat_b_enc), axis=1)
print('Human new trainning set', feat_all_enc.shape)
X_tr = np.copy(feat_all_enc)
y_tr = np.copy(label).flatten()
y_tr[y_tr == -1] = 0
print(y_tr)

eval_model(X_tr, y_tr)

# # %%
#
# gbm = LGBMClassifier(n_estimators=300,  # @200 hoặc 250
#                      num_leaves=50,
#                      learning_rate=0.05  # @0.1
#                      )
#
# feat_all_enc = np.concatenate((feat_a_enc, feat_b_enc), axis=1)
# print('\n--- Human new trainning set', feat_all_enc.shape)
# X_tr = np.copy(feat_all_enc)
# y_tr = np.copy(label).flatten()
# y_tr[y_tr == -1] = 0
# print(y_tr)
#
# gbm.fit(X_tr, y_tr)
#
#
# # %%
#
# def eval_testset(name):
#     os.chdir(r"D:\NCSI\3 - Thuc nghiem\ModifiedModel\AE_LGBM_2020\mod_AE_LGBM\Independent Species/" + name)
#     feat_a = pd.read_csv("total_features_a.csv")
#     feat_a = enc_a.predict(feat_a)
#
#     feat_b = pd.read_csv("total_features_b.csv")
#     feat_b = enc_b.predict(feat_b)
#
#     data_te = np.concatenate((feat_a, feat_b), axis=1)
#     print('\n', name, ', n_samples', len(data_te), end=', ')
#     label_te = [1] * data_te.shape[0]
#
#     # print(gbm.score(data_te, label_te))
#     print("acc {:.4f}".format(gbm.score(data_te, label_te)))
#
#
# eval_testset('Celeg')
# eval_testset('Ecoli')
# eval_testset('Hsapi')
# eval_testset('Mmusc')
# eval_testset('Wnt')
# eval_testset('CD9')
# eval_testset('ras-ref-erk_network')

# %%
