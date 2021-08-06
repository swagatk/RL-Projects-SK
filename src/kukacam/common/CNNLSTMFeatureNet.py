"""
Feature Network that combines CNN + LSTM
"""

from re import L
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import activations, layers


class CNNLSTMFeatureNetwork:
    def __init__(self, state_size: tuple, time_steps: int, 
                                        learning_rate=1e-3) -> None:
        self.state_size = state_size
        self.lr = learning_rate
        self.time_steps = time_steps

        # create model
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self, conv_layers=[16, 32, 32], 
                            dense_layers=[128, 128, 64]):
        
        input_shape = (self.time_steps, -1, ) + self.state_size
        
        org_input = tf.keras.layers.Input(shape=input_shape)
        loop_input = org_input 
        for i in range(len(conv_layers)):
            conv = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv2D(conv_layers[i], 
                            kernel_size=5, strides=2,
                            padding="SAME", activation="relu"))(loop_input)
            pool = tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4), 
                strides=None, padding="SAME"))(conv)
            loop_input = pool
        f = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten())(pool)
        f = tf.keras.layers.LSTM(time_steps, activations="relu", 
                            return_sequences=False)(f)
        for i in range(len(dense_layers)):
            f = tf.keras.layers.Dense(dense_layers[i], 
                                    activation="relu")(f)

        model = tf.keras.Model(inputs=org_input, outputs=f, name='feature_net')
        model.summary()
        keras.utils.plot_model(model, to_file='cnn_lstm_feature_net.png',
                        show_shapes=True, show_layer_names=True)
        return model 

    def __call__(self, state):
        # input is a tensor
        return self.model(state)




