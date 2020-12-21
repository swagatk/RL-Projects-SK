'''
Network for extracting features from input images
'''
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
###########################
# feature network
########################
class FeatureNetwork:
    def __init__(self, state_size, learning_rate=1e-3):
        self.state_size = state_size
        self.lr = learning_rate
        # create NN models
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        img_input = layers.Input(shape=self.state_size)

        # shared convolutional layers
        conv1 = layers.Conv2D(16, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(img_input)
        bn1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(bn1)
        bn2 = layers.BatchNormalization()(conv2)
        conv3 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(bn2)
        bn3 = layers.BatchNormalization()(conv3)
        f = layers.Flatten()(bn3)
        f = layers.Dense(128, activation="relu")(f)
        f = layers.Dense(128, activation="relu")(f)
        f = layers.Dense(64, activation="relu")(f)
        model = tf.keras.Model(inputs=img_input, outputs=f)
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)

