import pickle

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, LSTM, Bidirectional, Dense, Input, Embedding
from tensorflow.keras.initializers import GlorotUniform, RandomUniform
import tensorflow

np.random.seed(0)
tensorflow.random.set_seed(seed=0)


def one_hot(a, num_classes=20):
    a = np.array(a)
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def get_layer_embedding(w_eminit=RandomUniform(seed=1)):
    AA_1 = "ARNDCEQGHILKMFPSTWYV"
    aa_map = dict()
    for i, aa in enumerate(AA_1):
        aa_map[aa] = i

    input_V = one_hot(list(aa_map.values()), num_classes=20)
    layer = Embedding(
        input_dim=input_V.shape[0], output_dim=input_V.shape[1],
        embeddings_initializer=w_eminit,
        # weights=[input_V],
        # input_length=fix_len,
        trainable=True
    )
    return layer


# --- LUU
def bilstm(feature_dim, k_init=GlorotUniform(seed=0)):
    i = Input(shape=(1000,))
    x = get_layer_embedding()(i)
    x = Bidirectional(LSTM(feature_dim,
                           kernel_initializer=k_init,
                           return_sequences=False))(x)
    f = Flatten()(x)
    y = Dense(1024, kernel_initializer=k_init, activation='relu')(f)
    # x = Dense(100, kernel_initializer=k_init, activation='relu')(x)
    o = Dense(2, kernel_initializer=k_init, activation='softmax')(y)

    feat_model = Model(inputs=i, outputs=f)
    model = Model(inputs=i, outputs=o)
    return feat_model, model

# feat_model, model = bilstm(3605)
# print(feat_model.summary())
# print(model.summary())
