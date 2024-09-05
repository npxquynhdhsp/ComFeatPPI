"""
@thnhan
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, add, Concatenate
from tensorflow.keras.initializers import GlorotUniform

import tensorflow as tf


def module_feature_extraction(n_dim, W_regular, drop, n_units, kernel_init):
    dnn = Sequential()
    dnn.add(Dense(n_units, input_dim=n_dim,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 2,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 4,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 8,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    # dnn.add(Dense(n_units // 16,
    #               kernel_initializer=kernel_init,
    #               activation='relu',
    #               kernel_regularizer=W_regular))
    # dnn.add(BatchNormalization())
    # dnn.add(Dropout(drop))

    return dnn


def net(dim1, dim2, W_regular, drop=0.5, n_units=1024, seed=123456):
    """
    - dim1 = fixed protein length * word size, i.e. 557 * 20 = 11140
    - dim2 = 650
    """
    # ====== To reproduce
    tf.random.set_seed(seed)
    glouni = GlorotUniform(seed=seed)

    # ====== Extraction
    w1 = module_feature_extraction(dim1,
                                   W_regular=W_regular,
                                   drop=drop,
                                   n_units=n_units,
                                   kernel_init=glouni)
    s1 = module_feature_extraction(dim2,
                                   W_regular=W_regular,
                                   drop=drop,
                                   n_units=n_units,
                                   kernel_init=glouni)
    w2 = module_feature_extraction(dim1,
                                   W_regular=W_regular,
                                   drop=drop,
                                   n_units=n_units,
                                   kernel_init=glouni)
    s2 = module_feature_extraction(dim2,
                                   W_regular=W_regular,
                                   drop=drop,
                                   n_units=n_units,
                                   kernel_init=glouni)

    in1, in2 = Input(dim1), Input(dim2)
    in3, in4 = Input(dim1), Input(dim2)
    x1, x2 = w1(in1), s1(in2)
    x3, x4 = w2(in3), s2(in4)

    # ====== Merge 1, 2
    mer1 = add([x1, x2])
    den1 = Dense(5, kernel_initializer=glouni,
                 activation='relu',
                 kernel_regularizer=W_regular)(mer1)
    den1 = BatchNormalization()(den1)
    out1 = Dropout(drop)(den1)

    # ====== Merge 3, 4
    mer2 = add([x3, x4])  # Concatenate(axis=1)
    den2 = Dense(5, kernel_initializer=glouni,
                 activation='relu',
                 kernel_regularizer=W_regular)(mer2)
    den2 = BatchNormalization()(den2)
    out2 = Dropout(drop)(den2)

    # ====== Classification
    mer = add([out1, out2])
    y = Dense(4, kernel_initializer=glouni,
              activation='relu',
              kernel_regularizer=W_regular)(mer)
    y = BatchNormalization()(y)
    # y = Dropout(0.1)(y)
    out = Dense(2, kernel_initializer=glouni, activation='softmax')(y)

    final = Model(inputs=[in1, in2, in3, in4], outputs=out)
    # print(final.summary())
    tf.keras.utils.plot_model(final, "model.png", show_shapes=True)
    return final
