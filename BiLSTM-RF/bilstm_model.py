"""
reproduce: thnhan@hueuni.edu.vn
"""

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, LSTM, Bidirectional, Dense, Input, Embedding, Concatenate, \
    BatchNormalization
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
        trainable=True
    )
    return layer


def bilstm(feature_dim, k_init=GlorotUniform(seed=0)):
    ia = Input(shape=(515,))

    xa = get_layer_embedding()(ia)
    xa = Bidirectional(LSTM(feature_dim,
                            kernel_initializer=k_init,
                            return_sequences=True))(xa)
    fa = Flatten()(xa)

    ib = Input(shape=(515,))

    xb = get_layer_embedding()(ib)
    xb = Bidirectional(LSTM(feature_dim,
                            kernel_initializer=k_init,
                            return_sequences=True))(xb)

    fb = Flatten()(xb)

    f = Concatenate()([fa, fb])
    f = Dense(7210, kernel_initializer=k_init, activation='relu')(f)
    f = BatchNormalization()(f)
    y = Dense(1024, kernel_initializer=k_init, activation='relu')(f)
    y = BatchNormalization()(y)
    o = Dense(2, kernel_initializer=k_init, activation='softmax')(y)

    feat_model = Model(inputs=[ia, ib], outputs=f)
    model = Model(inputs=[ia, ib], outputs=o)
    return feat_model, model


if __name__ == "__main__":
    feat_model, model = bilstm(3605)
    print(feat_model.summary())
    print(model.summary())
