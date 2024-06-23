import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import LSTM, Dense, Input


def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True)(inputs)
    x = LSTM(50, return_sequences=False)(x)
    x = Dense(25)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
