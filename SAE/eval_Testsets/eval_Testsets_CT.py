import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from SAE_model import SAE_AC, SAE_CT
from feature.AC import AutoCovarianceEncoder
from feature.exAC import get_AC
from utils.fasta_tool import FASTA_to_ID_SEQ


def get_CT_testset(dset_name, pth):
    print(pth)
    A = pd.read_csv(pth + '/CT_A.csv', header=None)
    B = pd.read_csv(pth + '/CT_B.csv', header=None)
    return pd.concat([A, B], axis=1).values


def eval_testset_CT(args, dset_name, trained_model):
    os.chdir(args.testset + '/' + dset_name)
    print('\n---', os.getcwd())

    if not os.path.exists(args.testset + '/_LUU_CT/' + dset_name + '_pred.pkl'):
        # print('--- Test')
        te_X = get_CT_testset(dset_name, os.getcwd())
        te_y = np.array([1] * len(te_X))

        prob_y = trained_model.predict(te_X)
        pickle.dump(prob_y, open(args.testset + '/_LUU_CT/' + dset_name + '_pred.pkl', 'wb'))
        pred_y = np.argmax(prob_y, axis=1).flatten()
        # print(pred_y)
        print(dset_name, "acc {:.4f}".format(sum(pred_y == te_y) / len(te_y)))
    else:
        te_X = get_CT_testset(dset_name, os.getcwd())
        te_y = np.array([1] * len(te_X))
        prob_y = pickle.load(open(args.testset + '/_LUU_CT/' + dset_name + '_pred.pkl', 'rb'))
        pred_y = np.argmax(prob_y, axis=1).flatten()
        # print(pred_y)
        print(dset_name, "acc {:.4f}".format(sum(pred_y == te_y) / len(te_y)))


def eval_testsets_CT(args):
    os.chdir(args.yeastcore_dset)
    n_pos, n_neg = args.yeastcore_size

    if not os.path.exists('CT_A.csv'):
        print('RUN CT.R file in feature')
        exit(0)

    CT_a = pd.read_csv('CT_a.csv').values
    CT_b = pd.read_csv('CT_b.csv').values
    print(CT_a.shape, CT_b.shape)

    tr_X = np.concatenate([CT_a, CT_b], axis=1)
    tr_y = to_categorical([1] * n_pos + [0] * n_neg)

    model = SAE_CT(686)

    model.fit(tr_X, tr_y,
              epochs=args.epochs,
              batch_size=args.batch, verbose=2)

    eval_testset_CT(args, 'Celeg', model)
    eval_testset_CT(args, 'Ecoli', model)
    eval_testset_CT(args, 'Hsapi', model)
    eval_testset_CT(args, 'Hpylo', model)
    eval_testset_CT(args, 'Mmusc', model)
    eval_testset_CT(args, 'Dmela', model)

    eval_testset_CT(args, 'Wnt', model)
    eval_testset_CT(args, 'CD9', model)
    eval_testset_CT(args, 'Cancer_specific', model)

    return model
