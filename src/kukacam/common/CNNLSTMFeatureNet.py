"""
Feature Network that combines CNN + LSTM
"""

from collections import deque
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import activations, layers


class CNNLSTMFeatureNetwork:
    def __init__(self, state_size: tuple,  attn: dict, learning_rate=1e-3) -> None:
        self.state_size = state_size    # shape: 4 dim: stack_size, h, w, c
        self.lr = learning_rate
        self.step_counter = 0
        self.attn = attn
        self.stack_size = self.state_size[0]    # verify this ... 

        # create model
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self, conv_layers=[16, 32, 32], 
                            dense_layers=[128, 128, 64]):
        
        org_input = tf.keras.layers.Input(shape=self.state_size)
        x = org_input 
        for i in range(len(conv_layers)):
            x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv2D(conv_layers[i], 
                            kernel_size=5, strides=2,
                            padding="SAME", activation="relu"))(x)

            if self.attn is not None: 
                if self.attn['type'] == 'luong':
                    x = tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Attention())([x, x])
                elif self.attn['type'] == 'bahdanau':
                    x = tf.keras.layers.TimeDistributed(
                         tf.keras.layers.AdditiveAttention())([x, x])
                else:
                    raise ValueError('Wrong type of attention. Exiting ...')

            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                strides=None, padding="SAME"))(x)

        x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.LSTM(self.stack_size, activation="relu", 
                            return_sequences=False)(x)

        for i in range(len(dense_layers)):
            x = tf.keras.layers.Dense(dense_layers[i], 
                                    activation="relu")(x)

        model = tf.keras.Model(inputs=org_input, outputs=x, name='cnn_lstm_feature_net')
        model.summary()
        keras.utils.plot_model(model, to_file='cnn_lstm_feature_net.png',
                        show_shapes=True, show_layer_names=True)
        return model 

    def __call__(self, state):
        # input is a tensor of shape (-1, h, w, c)
        feature = self.model(state)
        return feature 

