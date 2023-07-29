import os

import numpy as np
import pandas as pd

from feature.AC import AutoCovarianceEncoder


def get_AC(dset):
    df = pd.read_csv(dset + '/uniprotein.txt', sep='\t')
    print(df)
    IDs, SEQs = df.id, df.protein
    print(len(SEQs))

    feats = []
    for prot in SEQs:
        feat = AutoCovarianceEncoder().to_feature(prot)
        feats.append(feat)

    print(len(feats))

    feats_df = pd.DataFrame(feats, index=IDs)
    print(feats_df)

    neg = pd.read_csv(dset + '/negative.txt', sep='\t')
    pos = pd.read_csv(dset + '/positive.txt', sep='\t')
    print(pos, neg)

    X_A = pd.concat([feats_df.loc[pos.proteinA], feats_df.loc[neg.proteinA]], axis=0)
    X_B = pd.concat([feats_df.loc[pos.proteinB], feats_df.loc[neg.proteinB]], axis=0)

    X_A.to_csv(dset + '/AC_a.csv', index=False)
    X_B.to_csv(dset + '/AC_b.csv', index=False)
    # X = np.concatenate([X_A, X_B], axis=1)
    print(X_A)
    print(X_B)

    return X_A.values, X_B.values
