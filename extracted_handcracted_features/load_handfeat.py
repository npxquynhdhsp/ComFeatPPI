"""
@author: thnhan
"""
import pickle

import pandas as pd
import numpy as np
from sys import path


def load_handfeat_YeastCore(dset_dir):
    # extracted_handcracted_features = pd.read_csv(dset_dir + '/extracted_handcracted_features.txt', index_col=0, header=None)
    handfeat = pickle.load(open(dset_dir + "/handfeat.pkl", "rb"))["handfeat"]
    pos = pd.read_csv(path[1] + '/datasets/Yeastcore/positive.txt', sep="\t")
    neg = pd.read_csv(path[1] + '/datasets/Yeastcore/negative.txt', sep="\t")
    pos_feat_A = handfeat.loc[pos.proteinA].values
    pos_feat_B = handfeat.loc[pos.proteinB].values
    neg_feat_A = handfeat.loc[neg.proteinA].values
    neg_feat_B = handfeat.loc[neg.proteinB].values
    return pos_feat_A, pos_feat_B, neg_feat_A, neg_feat_B


def load_handfeat_Human8161(dset_dir):
    # extracted_handcracted_features = pd.read_csv(dset_dir + '/extracted_handcracted_features.txt', index_col=0, header=None)
    handfeat = pickle.load(open(dset_dir + "/extracted_handcracted_features.pkl", "rb"))["extracted_handcracted_features"]
    pos = pd.read_csv(path[1] + '/datasets/Human8161/positive.txt', sep="\t")
    neg = pd.read_csv(path[1] + '/datasets/Human8161/negative.txt', sep="\t")
    pos_feat_A = handfeat.loc[pos.proteinA].values
    pos_feat_B = handfeat.loc[pos.proteinB].values
    neg_feat_A = handfeat.loc[neg.proteinA].values
    neg_feat_B = handfeat.loc[neg.proteinB].values
    return pos_feat_A, pos_feat_B, neg_feat_A, neg_feat_B


def FASTA_feat_to_numpy(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        inds = []
        if len(lines) > 0:
            for i, l in enumerate(lines):
                if l.startswith('>'):
                    inds.append(i)
            inds.append(len(lines))
            feat = []
            for i in range(len(inds) - 1):
                item = lines[inds[i]:inds[i + 1]]
                a = ''.join(item[1:]).replace('\n', '')
                a = a.strip("\n").split(",")
                feat.append([float(temp) for temp in a])
        else:
            print("====== FILE is EMPTY =======")
    return np.array(feat)


def load_handfeat_Yeastfull(dset_dir):
    pos_feat_A = FASTA_feat_to_numpy(dset_dir + '/feat_pos_A.txt')
    pos_feat_B = FASTA_feat_to_numpy(dset_dir + '/feat_pos_B.txt')
    neg_feat_A = FASTA_feat_to_numpy(dset_dir + '/feat_neg_A.txt')
    neg_feat_B = FASTA_feat_to_numpy(dset_dir + '/feat_neg_B.txt')
    return pos_feat_A, pos_feat_B, neg_feat_A, neg_feat_B


if __name__ == "__main__":
    # pos_feat_A, pos_feat_B, neg_feat_A, neg_feat_B = load_handfeat_Yeastcore("Yeastcore")
    # print(pos_feat_B.shape)
    # print(neg_feat_A.shape)
    # print(neg_feat_A)
    pos_feat_A, pos_feat_B, neg_feat_A, neg_feat_B = load_handfeat_Yeastfull("YeastFull")
    print(pos_feat_B.shape)
    print(neg_feat_A.shape)
    print(neg_feat_A)
