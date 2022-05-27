import numpy as np
import tensorflow as tf
from  keras.models import Model 
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
class Encoder():
    """
    A conv net that converts a given image stack into an 1-D feature vector
    """
    def __init__(self, obs_shape, feature_dim, 
                    conv_layers=[32, 32,],
                    dense_layers=[64,], 
                    filter_size=3):
        assert len(obs_shape) == 3 # (height, width, channels)
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.conv_filter_size = filter_size
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def _build_model(self):
        inputs = Input(shape=self.obs_shape, name="input_layer")
        x = inputs
        for i in range(len(self.conv_layers)):
            x = Conv2D(self.conv_layers[i], 
                        self.conv_filter_size,
                        strides=(1,1), activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
        x = GlobalAveragePooling2D()(x)
        x = tf.keras.layers.LayerNormalization()(x)
        for j in range(len(self.dense_layers)):
            x = Dense(self.dense_layers[j], activation='relu')(x)
        x = Dense(self.feature_dim, activation='linear')(x)
        outputs = tf.keras.layers.LayerNormalization(name="output_layer")(x)
        model = Model(inputs, outputs)
        return model

    def __call__(self, obs):
        # obs: a tensor or a numpy array
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
        # input: numpy arrays or tensors
        with tf.GradientTape() as tape:
            z_a = self.model(x_a)   # anchor
            z_p = self.model(x_p)   # positive
            loss = self.compute_loss(z_a, z_p)
        trainable_params = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(gradients, trainable_params))
        return loss 