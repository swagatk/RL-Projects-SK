"""
Feature Network that combines CNN + LSTM
"""

from collections import deque
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import activations, layers


class CNNLSTMFeatureNetwork:
    def __init__(self, state_size: tuple, stack_size: int, 
                                        learning_rate=1e-3) -> None:
        self.state_size = state_size
        self.lr = learning_rate
        self.stack_size = stack_size
        self.img_buffer = deque(maxlen=self.stack_size) 
        self.step_counter = 0

        # create model
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self, conv_layers=[16, 32, 32], 
                            dense_layers=[128, 128, 64]):
        
        input_shape = (self.stack_size, ) + self.state_size   
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
        f = tf.keras.layers.LSTM(self.stack_size, activation="relu", 
                            return_sequences=False)(f)
        for i in range(len(dense_layers)):
            f = tf.keras.layers.Dense(dense_layers[i], 
                                    activation="relu")(f)

        model = tf.keras.Model(inputs=org_input, outputs=f, name='cnn_lstm_feature_net')
        model.summary()
        keras.utils.plot_model(model, to_file='cnn_lstm_feature_net.png',
                        show_shapes=True, show_layer_names=True)
        return model 

    def __call__(self, state):
        # input is a tensor of shape (-1, h, w, c)
        stacked_img = self.prepare_input(state)
        return self.model(stacked_img)

    def prepare_input(self, state):
        # input : tensor of size (-1, h, w, c)
        temp_list = []
        for i in range(self.stack_size):
            if i < len(self.img_buffer):
                temp_list.append(self.img_buffer[-1-i])
            else:
                temp_list.append(state)

        stacked_img = tf.stack(temp_list, axis=1)
        return stacked_img      # check the shape: -1, stack_size, h, w, c
