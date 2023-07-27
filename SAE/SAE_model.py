"""
# # TÁI LẬP TRÌNH TỪ BÀI BÁO GỐC
# Sử dụng siêu tham số từ bài báo gốc
# Sử dụng chương trình trích xuất đặc trưng được cung cấp từ bài báo gốc
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow

from numpy.random import seed

seed(0)
tensorflow.random.set_seed(0)


def SAE_AC(input_dim, n_units=400, kernel_init=GlorotUniform(seed=0), drop=0.5, learning_rate=0.001):
    dnn = Sequential()

    dnn.add(Input(shape=(input_dim,)))

    dnn.add(Dense(n_units, kernel_initializer=kernel_init, activation='relu'))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(2, kernel_initializer=kernel_init, activation='softmax'))

    opt = Adam(learning_rate=learning_rate, decay=0.001)
    dnn.compile(optimizer=opt,
                loss=CategoricalCrossentropy(),
                metrics=['accuracy'])
    return dnn


def SAE_CT(input_dim, n_units=700, kernel_init=GlorotUniform(seed=0), drop=0.3, learning_rate=0.001):
    dnn = Sequential()

    dnn.add(Input(shape=(input_dim,)))

    dnn.add(Dense(n_units, kernel_initializer=kernel_init, activation='relu'))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(2, kernel_initializer=kernel_init, activation='softmax'))

    opt = Adam(learning_rate=learning_rate, decay=0.001)
    dnn.compile(optimizer=opt,
                loss=CategoricalCrossentropy(),
                metrics=['accuracy'])
    return dnn
