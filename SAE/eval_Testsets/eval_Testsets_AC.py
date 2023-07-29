import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from SAE_model import SAE_AC
from feature.AC import AutoCovarianceEncoder
from feature.exAC import get_AC
from utils.fasta_tool import FASTA_to_ID_SEQ


def get_AC_testset(dset_name, pth):
    print(dset_name)
    dset_A = pth + '/' + dset_name + '_ProA.txt'
    dset_B = dset_name + '_ProB.txt'

    if os.path.exists(dset_name + '_AC.npz'):
        return np.load(dset_name + '_AC.npz')['X']

    _, seq_A = FASTA_to_ID_SEQ(dset_A)
    _, seq_B = FASTA_to_ID_SEQ(dset_B)

    print(dset_name, len(seq_A))

    feat_A = []
    for prot in seq_A:
        feat = AutoCovarianceEncoder().to_feature(prot)
        feat_A.append(feat)

    feat_B = []
    for prot in seq_A:
        feat = AutoCovarianceEncoder().to_feature(prot)
        feat_B.append(feat)

    X = np.concatenate([feat_A, feat_B], axis=1)
    np.savez_compressed(dset_name + '_AC.npz', X=X)

    return X


def eval_testset_AC(args, dset_name, trained_model):
    os.chdir(args.testset + '/' + dset_name)
    print('\n---', os.getcwd())

    if not os.path.exists(args.testset + '/_LUU_AC/' + dset_name + '_pred.pkl'):
        print('--- Test')
        te_X = get_AC_testset(dset_name, os.getcwd())
        te_y = np.array([1] * len(te_X))

        prob_y = trained_model.predict(te_X)
        pickle.dump(prob_y, open(args.testset + '/_LUU_AC/' + dset_name + '_pred.pkl', 'wb'))
        pred_y = np.argmax(prob_y, axis=1).flatten()
        # print(pred_y)
        print(dset_name, "acc {:.4f}".format(sum(pred_y == te_y) / len(te_y)))
    else:
        te_X = get_AC_testset(dset_name, os.getcwd())
        te_y = np.array([1] * len(te_X))
        prob_y = pickle.load(open(args.testset + '/_LUU_AC/' + dset_name + '_pred.pkl', 'rb'))
        pred_y = np.argmax(prob_y, axis=1).flatten()
        # print(pred_y)
        print(dset_name, "acc {:.4f}".format(sum(pred_y == te_y) / len(te_y)))


def eval_testsets_AC(args):
    os.chdir(args.yeastcore_dset)
    n_pos, n_neg = args.yeastcore_size

    if not os.path.exists('AC_a.csv'):
        print('Wait')
        AC_a, AC_b = get_AC(args.yeastcore_dset)
    else:
        AC_a = pd.read_csv('AC_a.csv').values
        AC_b = pd.read_csv('AC_b.csv').values
        print(AC_a.shape, AC_b.shape)

    tr_X = np.concatenate([AC_a, AC_b], axis=1)
    tr_y = to_categorical([1] * n_pos + [0] * n_neg)

    model = SAE_AC(420)

    model.fit(tr_X, tr_y,
              epochs=args.epochs,
              batch_size=args.batch, verbose=2)

    eval_testset_AC(args, 'Celeg', model)
    eval_testset_AC(args, 'Ecoli', model)
    eval_testset_AC(args, 'Hsapi', model)
    eval_testset_AC(args, 'Hpylo', model)
    eval_testset_AC(args, 'Mmusc', model)
    eval_testset_AC(args, 'Dmela', model)

    eval_testset_AC(args, 'Wnt', model)
    eval_testset_AC(args, 'CD9', model)
    eval_testset_AC(args, 'Cancer_specific', model)

    return model
