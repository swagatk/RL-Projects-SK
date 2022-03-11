import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Lambda
from  tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D
from torch import mode

class Encoder():
    def __init__(self, obs_shape, feature_dim, num_layers=2,
                        num_filters=32):
        assert len(obs_shape) == 3 # (height, width, channels)
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        inputs = Input(shape=obs_shape)
        x = inputs
        for _ in range(num_layers - 1):
            x = Conv2D(num_filters, (3, 3), stride=1, activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
        x = GlobalAveragePooling2D()(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = Dense(feature_dim * num_filters, activation='relu')(x)
        outputs = Dense(feature_dim)(x)
        model = Model(inputs, outputs)
        return model

    def reparameterize(self, mu, logstd):
        std = tf.exp(logstd)
        eps = tf.random.normal(std.shape)
        return mu + eps * std


class SiameseNetwork():
    def __init__(self, obs_shape, z_dim, embedding_dim) -> None:
        """
        Args:
            obs_shape: tuple of ints
            z_dim: int (bilinear reparameterization trick)
            embedding_dim: int (feature dim of the encoder)
        """
        assert obs_shape.ndim == 3      # (height, width, channels)
        self.query_encoder = Encoder(obs_shape, embedding_dim)
        self.key_encoder = Encoder(obs_shape, embedding_dim)
        self.W = tf.Variable(tf.random.uniform(shape=(z_dim, z_dim), minval=-0.1, maxval=0.1))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def encode(self, obs):
        """ encode the observation into a latent space 
        obs: tensor of shape: (batch_size, height, width, channels)

        """
        query_embedding = self.query_encoder(obs)
        key_embedding = self.key_encoder(obs)
        return query_embedding, key_embedding

    def compute_logits(self, z_k, z_q):
        proj_k = tf.linalg.matmul(self.W, tf.transpose(z_k)) # (z_dim, B)
        logits = tf.linalg.matmul(z_q, proj_k) # (B, B)
        logits = logits - tf.reduce_max(logits, axis=1) # for stability
        labels = tf.range(logits.shape[0])
        return logits, labels # (B, B)

    def train(self, data):
        """ train the model on the data
        data: tuple (obs & augmented obs): obs tensor of shape: (batch_size, height, width, channels) 
        """
        x_q, x_k = data 

        with tf.GradientTape() as tape:
            z_q = self.query_encoder(x_q)
            z_k = tf.stop_gradient(self.key_encoder(x_k))
            logits, labels = self.compute_logits(z_k, z_q)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        gradients = tape.gradient(loss, self.W)
        self.optimizer.apply_gradients(zip(gradients, [self.W]))




    




    