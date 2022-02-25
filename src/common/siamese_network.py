import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Lambda
from  tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D
from torch import mode

class SiameseNetwork():
    def __init__(self, obs_shape, feature_dim, num_layers=2,
                        num_filters=32) -> None:
        assert len(obs_shape) == 3 # (height, width, channels)
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        inputs = Input(shape=obs_shape)
        x = inputs
        for _ in range(num_layers - 1):
            x = Conv2D(num_filters, (3, 3), stride=1, activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
        x = GlobalAveragePooling2D()(x)
        x = tf.keras.layers.LayerNormalization()(x)
        outputs = Dense(feature_dim)(x)
        model = Model(inputs, outputs)
        return model

    def reparameterize(self, mu, logstd):
        std = tf.exp(logstd)
        eps = tf.random.normal(std.shape)
        return mu + eps * std