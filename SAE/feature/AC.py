"""
Auto Covariance Descriptors is a statistical tool proposed by Wold et al.
Protein protein --> 210-dimensional representation vector

@coding: thnhan
"""
import os

import numpy as np
import pandas as pd
from sys import path
from utils import rootdir


# from protein_utils.AA import AA_1


def normalized(supp=None):
    if supp is None:
        supp = r'D:\NCSI\SAE_code\feature\AutoCovariance.csv'

    physical = pd.read_csv(supp, index_col=0)
    # physical.drop(columns=['M'], inplace=True)
    m = physical.mean()
    s = physical.std()
    col = physical.columns
    idx = physical.index
    physical = physical.values.transpose()
    physical = (physical - m.values.reshape(7, 1)) / s.values.reshape(7, 1)
    physical = pd.DataFrame(data=physical.T, columns=col, index=idx)
    return physical


def check_protein(sequence):
    for aa in sequence:
        if aa not in AA_1:
            return False
    return True


class AutoCovarianceEncoder:
    def __init__(self, lg=30):
        self.normalized_table = normalized()
        self.minLength = 31
        self.dim = 7 * lg  # self.get_dimensional()
        self.shortName = 'AC'
        self.fullName = 'Auto Covariance'

    def to_feature(self, sequence, lg=30):
        """ lg = lambda """
        # Loại bỏ AA không xác định "tr_X_AC", "U", và "B"
        # re.sub("XBU", sequence)
        sequence = sequence.replace('X', '')
        sequence = sequence.replace('U', '')
        sequence = sequence.replace('B', '')

        if len(sequence) <= lg:
            return np.zeros(shape=7 * lg)

        L = len(sequence)
        P = self.normalized_table.loc[list(sequence)]

        mean_values = P.mean(axis=0)
        features = np.zeros(7 * lg)

        index = 0
        for j in self.normalized_table.columns:
            t = P[j] - mean_values[j]
            for lag in range(1, lg + 1):
                x1 = t.iloc[:L - lag]
                x2 = t.iloc[lag:]
                AC = np.dot(x1.values, x2.values.T)
                AC *= 1 / (L - lag)
                features[index] = AC
                index += 1

        return features


AA_1 = "ARNDCEQGHILKMFPSTWYV"
