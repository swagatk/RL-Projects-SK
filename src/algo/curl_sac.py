'''
CURL implementation in Tensorflow 2.0
Source for Pytorch implementation: 
URL: https://github.com/MishaLaskin/curl
'''
import numpy as np
import tensorflow as tf

from common.siamese_network import Encoder, SiameseNetwork

class curlActor():
    """ MLP Actor Network"""
    def __init__(self, obs_shape, action_shape, feature_dim, hidden_dim,
                    log_std_min=-20, log_std_max=2, encoder_model=None):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder = encoder_model
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.obs_shape)
        if self.encoder is None:
            x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(inputs)
        else:
            x = self.encoder(inputs)
        x = tf.keras.layers.Dense(self.hidden_dim*2, activation='relu')(x)
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x)
        mu = tf.keras.layers.Dense(self.action_shape[0], activation='tanh')(x)
        log_sig = tf.keras.layers.Dense(self.action_shape[0])(x)
        model = tf.keras.Model(inputs=inputs, outputs=[mu, log_sig])
        return model 

    
    def __call__(self, obs):
        # input is a tensor
        mu, log_sigma = self.model(obs)
        std = tf.math.exp(log_sigma)
        return mu, std


class curlCritic():
    def __init__(self, obs_shape, action_shape, gamma, feature_model=None) -> None:
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.feature_model = feature_model
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.obs_shape)
        if self.feature_model is None:
            x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(inputs)
        else:
            x = self.feature_model(inputs)
        x = tf.keras.layers.Dense(self.hidden_dim*2, activation='relu')(x)
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x)
        v = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=v)
        return model
