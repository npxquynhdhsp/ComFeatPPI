import os
import pickle

import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import StratifiedKFold
from SAE_model import SAE_AC, SAE_CT
from feature.exAC import get_AC
from utils.report_result import print_metrics, my_cv_report
from tensorflow.keras.callbacks import ModelCheckpoint

from utils.rmalldir import rmalldir


def eval_Yeastcore_AC(args):
    os.chdir(args.yeastcore_dset)
    n_pos, n_neg = args.yeastcore_size

    print("\n---", os.getcwd())

    if not os.path.exists('AC_a.csv'):
        print('Wait')
        AC_a, AC_b = get_AC(args.yeastcore_dset)
    else:
        AC_a = pd.read_csv('AC_a.csv').values
        AC_b = pd.read_csv('AC_b.csv').values
        print(AC_a.shape, AC_b.shape)

    X = np.concatenate([AC_a, AC_b], axis=1)
    y = np.array([1] * n_pos + [0] * n_neg)

    skf = StratifiedKFold(n_splits=args.validation, random_state=48, shuffle=True)

    scores, cv_prob_Y, cv_test_y = [], [], []
    method_result = dict()

    for ii, (tr_ii, te_ii) in enumerate(skf.split(X, y)):
        tr_X, te_X = X[tr_ii], X[te_ii]
        tr_y, te_y = to_categorical(y[tr_ii]), to_categorical(y[te_ii])

        model = SAE_AC(420)

        # #####################################################
        checkpoint_filepath = './tmp/checkpoint'  # chu y: dat '.' truoc '/'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        model.fit(tr_X, tr_y,
                  validation_data=(te_X, te_y),
                  callbacks=[model_checkpoint_callback],
                  epochs=args.epochs,
                  batch_size=args.batch, verbose=2)
        model.load_weights(checkpoint_filepath)

        rmalldir('./tmp')
        # #####################################################

        prob_y = model.predict(te_X)

        method_result['fold' + str(ii)] = {"true": te_y[:, 1], "prob": prob_y}
        pickle.dump(method_result, open("SAE_eval_Yeastcore.pkl", "wb"))

        scr = print_metrics(te_y[:, 1], prob_y[:, 1])
        scores.append(scr)

        cv_prob_Y.append(prob_y)
        cv_test_y.append(te_y[:, 1])

    # ====== FINAL REPORT
    print("\nFinal scores (mean)")
    scores_array = np.array(scores)
    my_cv_report(scores_array)


def eval_Yeastcore_CT(args):
    os.chdir(args.yeastcore_dset)
    n_pos, n_neg = args.yeastcore_size

    print("\n---", os.getcwd())

    if not os.path.exists('CT_a.csv'):
        print("plz RUN file CT.R first")
        exit(1)
    CT_a = pd.read_csv('CT_a.csv').values
    CT_b = pd.read_csv('CT_b.csv').values
    print(CT_a.shape, CT_b.shape)

    X = np.concatenate([CT_a, CT_b], axis=1)
    y = np.array([1] * n_pos + [0] * n_neg)

    skf = StratifiedKFold(n_splits=args.validation, random_state=48, shuffle=True)

    scores, cv_prob_Y, cv_test_y = [], [], []
    method_result = dict()

    for ii, (tr_ii, te_ii) in enumerate(skf.split(X, y)):
        tr_X, te_X = X[tr_ii], X[te_ii]
        tr_y, te_y = to_categorical(y[tr_ii]), to_categorical(y[te_ii])

        model = SAE_CT(686)

        # #####################################################
        checkpoint_filepath = './tmp/checkpoint'  # chu y: dat '.' truoc '/'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        model.fit(tr_X, tr_y,
                  validation_data=(te_X, te_y),
                  callbacks=[model_checkpoint_callback],
                  epochs=args.epochs,
                  batch_size=args.batch, verbose=2)
        model.load_weights(checkpoint_filepath)

        rmalldir('./tmp')
        # #####################################################

        prob_y = model.predict(te_X)

        # print(prob_y)
        method_result['fold' + str(ii)] = {"true": te_y[:, 1], "prob": prob_y}
        pickle.dump(method_result, open("SAE_eval_Yeastcore.pkl", "wb"))

        scr = print_metrics(te_y[:, 1], prob_y[:, 1])
        scores.append(scr)

        cv_prob_Y.append(prob_y)
        cv_test_y.append(te_y[:, 1])

    # ====== FINAL REPORT
    print("\nFinal scores (mean)")
    scores_array = np.array(scores)
    my_cv_report(scores_array)
