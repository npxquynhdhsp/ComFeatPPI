# %%
import os

import pandas as pd

from utils.fasta import fasta_to_dataframe

# neg_A = fasta_to_dataframe("neg_A.txt").reset_index()
# print(neg_A)
#
# neg_B = fasta_to_dataframe("neg_B.txt").reset_index()
# print(neg_B)
#
# neg = pd.concat([neg_A.protein, neg_B.protein], axis=1)
# neg.columns = ['chain', 'chain.1']
# print(neg)
#
# neg.to_csv('Negative YeastFull PPI.csv', index=False)

# %%


os.chdir(r"D:\NCSI\3 - Thuc nghiem\ModifiedModel\code_PPI_new_OK\AE_LGBM_2020, full paper\AE_LGBM_2020\mod_AE_LGBM\Independent Species\Hpylo")
pos_A = fasta_to_dataframe("Hpylo_ProA.txt").reset_index()
print(pos_A)

pos_B = fasta_to_dataframe("Hpylo_ProB.txt").reset_index()
print(pos_B)

pos = pd.concat([pos_A.protein, pos_B.protein], axis=1)
pos.columns = ['chain', 'chain.1']
print(pos)
pos.to_csv('Hpylo_PPI.csv', index=False)

# %%

