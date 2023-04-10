import numpy as np

from bilstm_rf import bilstm
from utils.dset_tool import load_raw_dset


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
        # print(embedd)
        if len(embedd) < fixlen:
            o = np.array([25] * (fixlen - len(embedd)))
            embedd = np.hstack([embedd, o])
        else:
            embedd = embedd[:fixlen]
        embedd_lst.append(embedd)
    print(np.array(embedd_lst).shape)
    return np.array(embedd_lst)


def prepare_data(inp_prot_lst):
    # data = []
    #
    # # prot_subs = []
    # # for prot in inp_prot_lst:
    # #     prot_subs.append(prot[:l_subs])
    # # data.append(get_embedding_lst(prot_subs, l_subs))
    # # for i in range(2, n_subs + 1):
    # #     prot_subs = []
    # #     for prot in inp_prot_lst:
    # #         prot_subs.append(prot[(i - 1) * l_subs - 250: i * l_subs])  # 150
    # #     data.append(get_embedding_lst(prot_subs, l_subs))
    #
    # # --- đặc trưng nguyên dãy
    # data.append()
    # # ---
    return get_embedding_lst_token(inp_prot_lst, fix_len)


# %%


fix_len = 500
print("\n--- Prepare Dataset ...", end=" ")
dset, infor = load_raw_dset("data/Yeastcore")
P_seq_A, P_seq_B, N_seq_A, N_seq_B = dset['seq_pairs']
labels = dset['labels']

# X_inds = np.arange(len(labels))

e_PA = prepare_data(P_seq_A)
e_PB = prepare_data(P_seq_B)
e_NA = prepare_data(N_seq_A)
e_NB = prepare_data(N_seq_B)

print("Done")
print(e_PA)

# %%
p = np.concatenate([e_PA, e_PB], axis=1)
n = np.concatenate([e_NA, e_NB], axis=1)
X = np.concatenate([p, n], axis=0)
print(X.shape)
del p, n

# %%


print(e_PA)
print(labels)

feat_model, model = bilstm(3605)
print(feat_model.summary())
print(model.summary())

# %%

ind = np.arange(len(labels))
np.shuffle(ind)
X, y = X[ind], labels[ind]

model.compile(optimizer='adam', loss='catelogical_crossentropy', metrics='accuracy')

model.fit(X, y, epochs=50, batch_size=128)

model.save('full_bilstm.h5')
feat_model.save('encoder_bilstm.h5')
