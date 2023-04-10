# %%
import pickle
import time

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from utils.report_result import print_metrics, my_cv_report



import os

os.chdir(r"D:\NCSI\3 - Thuc nghiem\ModifiedModel\AE_LGBM_2020\mod_AE_LGBM\_original\Human")

label = pd.read_csv("label.csv")
feat_a = pd.read_csv("total_features_a.csv")
feat_b = pd.read_csv("total_features_b.csv")
feat_all = pd.concat([feat_a, feat_b], axis=1)
print(feat_all.shape)

# %%
from tensorflow.keras.models import load_model
enc_a = load_model("AE_Human_a_.h5")
print(enc_a.summary())

X_no_enc = feat_all.values
y_no_enc = label.values

feat_a_enc = enc_a.predict(feat_a)
print(feat_a_enc.shape)

enc_b = load_model("AE_Human_b_.h5")
print(enc_b.summary())

feat_b_enc = enc_b.predict(feat_b)
print(feat_b_enc.shape)

# %%
import numpy as np

feat_all_enc = np.concatenate((feat_a_enc, feat_b_enc), axis=1)
print('Human new trainning set', feat_all_enc.shape)

# %%
import numpy as np

X_tr = np.copy(feat_all_enc)
y_tr = np.copy(label).flatten()
y_tr[y_tr == -1] = 0
print(y_tr)

# %%
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

#
#
# # gbm = lgb.LGBMClassifier(**params)
#
# # pipeline = Pipeline([('transformer', t), ('estimator', gbm)])
# cv = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
# print(f'\n{cv.__str__()}...')
# scr = cross_val_score(estimator=gbm, scoring=['auc'], X=X_tr, y=y_tr, cv=cv, n_jobs=2)


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

        model = lgb.LGBMClassifier(
            n_estimators=300,  # 500
            num_leaves=80,  # 80
            learning_rate=0.05
            # max_depth=10,
            # feature_fraction=0.9,
            # bagging_fraction=0.8,
            # bagging_freq=5,
        )

        model.fit(tr_X, tr_y)

        # # ====== SAVE MODEL
        # model.save("trained/model_trained_on_Yeastcore_fold" + str(i) + ".h5")

        # ====== REPORT
        prob_y = model.predict(te_X)

        y_true_prob['fold' + str(i)] = {"true_y": te_y, "prob_y": prob_y}
        pickle.dump(y_true_prob, open(r'AE_Human_y_true_prob.pkl', 'wb'))
        # ======

        scr = print_metrics(te_y, prob_y)
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

eval_model(X_tr, y_tr)


#
#
#
# # %%
# print(scr)
# print("mean {:0.4f}, std {:0.3f}".format(np.mean(scr), np.std(scr)))
#
# # %%
# import lightgbm as lgb
#
# gbm = lgb.LGBMClassifier(
#     n_estimators=500,
#     num_leaves=80,
#     learning_rate=0.05
# )
#
# gbm.fit(X_tr, y_tr.ravel())
#
# # %% [markdown]
# # # TEST TRÊN TẬP ĐỘC LẬP
#
# # %%
# import os
# os.chdir(r"D:\NCSI\3 - Thuc nghiem\ModifiedModel\AE_LGBM_2020\mod_AE_LGBM\Independent Species\Ecoli")
# feat_a = pd.read_csv("total_features_a.csv")
# feat_a = enc_a.predict(feat_a)
#
# feat_b = pd.read_csv("total_features_b.csv")
# feat_b = enc_b.predict(feat_b)
#
# data_te = np.concatenate((feat_a, feat_b), axis=1)
# print('data_te', data_te.shape)
# label_te = [1] * data_te.shape[0]
#
# # %%
# print(gbm.score(data_te, label_te))
# print("{:.4f}".format(gbm.score(data_te, label_te)))
#
# # %%
#



# %%

