"""
reproduce: thnhan@hueuni.edu.vn
"""
import os

import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from utils.dset_tool import load_raw_dset
from utils.fasta_tool import get_protein_from_fasta
from utils.report_result import print_metrics, my_cv_report
from train_bilstm_model import train_bilstm


def eval_Testsets(trained_bilstm, trained_rf, dset_name):
    os.chdir('D:\BiLSTM-RF\data\Testsets/' + dset_name)

    P_seq_A = get_protein_from_fasta(dset_name + '_proA.txt')
    P_seq_B = get_protein_from_fasta(dset_name + '_proA.txt')

    te_a = prepare_data(P_seq_A)
    te_b = prepare_data(P_seq_B)
    te_y = np.array([1] * len(te_a))

    te_X = trained_bilstm.predict([te_a, te_b])
    prob_y = trained_rf.predict_proba(te_X)

    pickle.dump(prob_y, open(r'D:\BiLSTM-RF\data\Testsets/' + dset_name + '_pred.pkl', 'wb'))
    pred_y = np.argmax(prob_y, axis=1).flatten()
    # print(pred_y)
    print(dset_name, "acc {:.4f}".format(sum(pred_y == te_y) / len(te_y)))


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
            o = np.array([20] * (fixlen - len(embedd)))
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
dset, infor = load_raw_dset("data/Yeastcore")
P_seq_A, P_seq_B, N_seq_A, N_seq_B = dset['seq_pairs']
labels = dset['labels']

e_PA = prepare_data(P_seq_A)
e_PB = prepare_data(P_seq_B)
e_NA = prepare_data(N_seq_A)
e_NB = prepare_data(N_seq_B)

a = np.concatenate([e_PA, e_NA], axis=0)
b = np.concatenate([e_PB, e_NB], axis=0)

bilstm, _ = train_bilstm(a, b, labels)

tr_X = bilstm.predict([a, b])
print(tr_X.shape)

# rf_cls = XGBClassifier(n_estimators=100).fit(tr_X, labels)

# %%

rf_cls = RandomForestClassifier(500).fit(tr_X, labels)

# %%

# eval_Testsets(bilstm, rf_cls, 'Celeg')
# eval_Testsets(bilstm, rf_cls, 'Dmela')
# eval_Testsets(bilstm, rf_cls, 'Ecoli')
# eval_Testsets(bilstm, rf_cls, 'Hpylo')
# eval_Testsets(bilstm, rf_cls, 'Hsapi')
# eval_Testsets(bilstm, rf_cls, 'Mmusc')
#
# # %%
#
# eval_Testsets(bilstm, rf_cls, 'CD9')
# eval_Testsets(bilstm, rf_cls, 'Wnt')
eval_Testsets(bilstm, rf_cls, 'Cancer_specific')

# %%

# Celeg acc 0.8522
# Dmela acc 0.8358
# Ecoli acc 0.7466
# Hpylo acc 0.7380
# Hsapi acc 0.8067
# Mmusc acc 0.8307
# CD9 acc 1.0000
# Wnt acc 0.8646
# Cancer_specific acc 0.9074
