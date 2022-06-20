import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Lambda
from  tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D

class Encoder():
    def __init__(self, obs_shape, feature_dim, num_layers=2,
                        num_filters=32):
        assert len(obs_shape) == 3 # (height, width, channels)
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def _build_model(self):
        inputs = Input(shape=self.obs_shape)
        x = inputs
        for _ in range(self.num_layers - 1):
            x = Conv2D(self.num_filters, (3, 3), strides=(1,1), activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
        x = GlobalAveragePooling2D()(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = Dense(self.feature_dim * self.num_filters, activation='relu')(x)
        outputs = Dense(self.feature_dim)(x)
        model = Model(inputs, outputs)
        return model

    def reparameterize(self, mu, logstd):
        std = tf.exp(logstd)
        eps = tf.random.normal(std.shape)
        return mu + eps * std

    def __call__(self, obs):
        # obs: a tensor
        f = self.model(obs)
        return f 

    def compute_loss(self, p, z):
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)
        # maximize cosine similarity
        loss = -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))
        return loss 

    def train(self, x_a, x_p):
        # can it handle input numpy arrays
        # x_a_tensor = tf.convert_to_tensor(x_a, dtype=np.float32)
        # x_p_tensor = tf.convert_to_tensor(x_p, dtype=np.float32)
        with tf.GradientTape() as tape:
            z_a = self.model(x_a)
            z_p = self.model(x_p)
            loss = self.compute_loss(z_a, z_p)
        trainable_params = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(gradients, trainable_params))
        return loss 



class SiameseNetwork():
    def __init__(self, obs_shape, z_dim)-> None:
        """
        Args:
            obs_shape: tuple of ints
            z_dim: int (bilinear reparameterization trick)
            embedding_dim: int (feature dim of the encoder)
        """
        assert len(obs_shape) == 3, '3D observation'      # (height, width, channels)
        self.query_encoder = Encoder(obs_shape, z_dim)
        self.key_encoder = Encoder(obs_shape, z_dim)
        self.W = tf.Variable(tf.random.uniform(shape=(z_dim, z_dim), 
                                        minval=-0.1, maxval=0.1))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # initially query and key encoders share same weights
        self.key_encoder.model.set_weights(self.query_encoder.model.get_weights())

    def encode(self, obs):
        """ encode the observation into a latent space 
        obs: tensor of shape: (batch_size, height, width, channels)

        """
        query_embedding = self.query_encoder(obs)
        key_embedding = self.key_encoder(obs)

        return query_embedding, key_embedding

    def compute_logits(self, z_q, z_k):
        proj_k = tf.linalg.matmul(self.W, tf.transpose(z_k)) # (z_dim, B)
        logits = tf.linalg.matmul(z_q, proj_k) # (B, B)
        logits = logits - tf.reduce_max(logits, axis=1) # for stability
        labels = tf.range(logits.shape[0])
        return logits, labels # (B, B)

    def train(self, data):
        """ train the model on the data
        data: tuple (obs & augmented obs): obs tensor of shape: (batch_size, height, width, channels) 
        """
        x_q, x_k = data # anchor, positives
        with tf.GradientTape() as tape1:
            z_q = tf.stop_gradient(self.query_encoder(x_q))   # anchor 
            z_k = tf.stop_gradient(self.key_encoder(x_k))    # positives
            logits, labels = self.compute_logits(z_q, z_k)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, axis=-1)
            train_wts = [self.W]
        gradients_1 = tape1.gradient(loss, train_wts)
        self.optimizer.apply_gradients(zip(gradients_1, train_wts))

        with tf.GradientTape() as tape2:
            z_q = self.query_encoder(x_q)    # anchor 
            z_k = tf.stop_gradient(self.key_encoder(x_k)) # positives
            logits, labels = self.compute_logits(z_q, z_k)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            enc_wts = self.query_encoder.model.trainable_variables
        gradients_2 = tape2.gradient(loss, enc_wts)
        self.optimizer.apply_gradients(zip(gradients_2, enc_wts))

        return loss 

    def update_key_encoder_wts(self, tau=0.995):
        for theta_key, theta_query in zip(self.key_encoder.model.trainable_variables,
                    self.query_encoder.model.trainable_variables):
            theta_key = tau * theta_key + (1 - tau) * theta_query


    




    