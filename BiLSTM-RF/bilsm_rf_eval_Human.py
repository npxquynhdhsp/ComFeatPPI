"""
reproduce: thnhan@hueuni.edu.vn
"""

import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from utils.dset_tool import load_raw_dset
from utils.report_result import print_metrics, my_cv_report
from train_bilstm_model import train_bilstm


def eval_Human(X_a, X_b, y):
    skf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)

    scores, cv_prob_Y, cv_test_y = [], [], []
    method_result = dict()

    for ii, (tr_ii, te_ii) in enumerate(skf.split(X_a, y)):
        tr_bilsm_a, tr_bilsm_b, tr_y = X_a[tr_ii], X_b[tr_ii], y[tr_ii]
        te_bilsm_a, te_bilsm_b, te_y = X_a[te_ii], X_b[te_ii], y[te_ii]

        bilstm, _ = train_bilstm(tr_bilsm_a, tr_bilsm_b, tr_y)

        tr_X = bilstm.predict([tr_bilsm_a, tr_bilsm_b])
        print(tr_X.shape)
        te_X = bilstm.predict([te_bilsm_a, te_bilsm_b])

        # --- RandomForestClassifier
        rf_cls = RandomForestClassifier(100)

        rf_cls.fit(tr_X, tr_y)
        prob_y = rf_cls.predict_proba(te_X)

        method_result['fold' + str(ii)] = {"true": te_y, "prob": prob_y}
        pickle.dump(method_result, open("BiLSTM_RF_eval_Human.pkl", "wb"))

        scr = print_metrics(te_y, prob_y[:, 1])
        scores.append(scr)

        cv_prob_Y.append(prob_y[:, 1])
        cv_test_y.append(te_y)

    # ====== FINAL REPORT
    print("\nFinal scores (mean)")
    scores_array = np.array(scores)
    my_cv_report(scores_array)


def get_embeding_prot_token(inp_prot):
    AA_1 = "ARNDCEQGHILKMFPSTWYV"
    aa_map = dict()
    for i, aa in enumerate(AA_1):
        aa_map[aa] = i

    inp_prot = inp_prot.replace("X", "")
    inp_prot = inp_prot.replace("U", "")
    inp_prot = inp_prot.replace("B", "")
    inp_prot = inp_prot.replace("Z", "")
    inp_prot = inp_prot.replace("O", "")

    prot_new = [aa_map[aa] for aa in inp_prot]
    return np.array(prot_new)


def get_embedding_lst_token(lst, fixlen):
    embedd_lst = []
    for i, prot in enumerate(lst):
        embedd = get_embeding_prot_token(prot)
        if len(embedd) < fixlen:
            o = np.array([25] * (fixlen - len(embedd)))
            embedd = np.hstack([embedd, o])
        else:
            embedd = embedd[:fixlen]
        embedd_lst.append(embedd)
    return np.array(embedd_lst)


def prepare_data(inp_prot_lst):
    return get_embedding_lst_token(inp_prot_lst, fix_len)


# %%

fix_len = 515
print("\n--- Prepare Dataset ...", end=" ")
dset, infor = load_raw_dset("data/Human")
P_seq_A, P_seq_B, N_seq_A, N_seq_B = dset['seq_pairs']
labels = dset['labels']

e_PA = prepare_data(P_seq_A)
e_PB = prepare_data(P_seq_B)
e_NA = prepare_data(N_seq_A)
e_NB = prepare_data(N_seq_B)

a = np.concatenate([e_PA, e_NA], axis=0)
b = np.concatenate([e_PB, e_NB], axis=0)

eval_Human(a, b, labels)
