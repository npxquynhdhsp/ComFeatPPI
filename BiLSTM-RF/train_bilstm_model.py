"""
reproduce: thnhan@hueuni.edu.vn
"""
from tensorflow.keras.utils import plot_model

from bilstm_model import bilstm
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


def train_bilstm(X_a, X_b, y):
    encoder_model, model = bilstm(7)
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics='accuracy')
    model.fit([X_a, X_b], to_categorical(y), epochs=30, batch_size=128)
    plot_model(model, to_file='model.png', show_shapes=True)
    plot_model(encoder_model, to_file='encoder.png', show_shapes=True)
    model.save('full_bilstm.h5')
    # feat_model.save('encoder_bilstm.h5')
    return encoder_model, model


def load_bilstm():
    e = load_model("encoder_bilstm.h5")
    print(e.summary())
    return e
