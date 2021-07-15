'''
Network for extracting features from input images
'''
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# local import
###########################
# feature network
########################
class FeatureNetwork:
    def __init__(self, state_size, learning_rate=1e-3):
        self.state_size = state_size
        self.lr = learning_rate
        # create NN models
        self.model = self._build_net()
        #self.model = self._build_net2()
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
        model = tf.keras.Model(inputs=img_input, outputs=f, name='feature_net')
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def _build_net2(self):
        img_input = layers.Input(shape=self.state_size)

        # shared convolutional layers
        conv1 = layers.Conv2D(64, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(img_input)
        p1 = layers.MaxPooling2D(pool_size=(4, 4), strides=None, padding='SAME')(conv1)
        conv2 = layers.Conv2D(128, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(p1)
        p2 = layers.MaxPooling2D(pool_size=(4, 4), strides=None, padding='SAME')(conv2)
        conv3 = layers.Conv2D(128, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(p2)
        p3 = layers.MaxPooling2D(pool_size=(4, 4), strides=None, padding='SAME')(conv3)
        conv4 = layers.Conv2D(64, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(p3)
        p4 = layers.MaxPooling2D(pool_size=(4, 4), strides=None, padding='SAME')(conv4)
        f = layers.Flatten()(p4)
        f = layers.Dense(256, activation="relu")(f)
        f = layers.Dense(256, activation="relu")(f)
        f = layers.Dense(128, activation="relu")(f)
        f = layers.Dense(64, activation="relu")(f)
        model = tf.keras.Model(inputs=img_input, outputs=f, name='feature_net_2')
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)


##################################
'''
attention layer is applied in between conv layers. Typical architecture looks like:
Conv2d - Attention - Conv2d - Attention - Conv2D - Attention - Flatten - Dense Layers

attn_type: 'bahdanau', 'luong'
arch = 1: attention(x) in-between conv layers
arch = 2: x + attention(x) in-between conv layers
arch = 3: x * attention(x) in-between conv layers

x is the output from previous conv layer
'''


class AttentionFeatureNetwork:
    def __init__(self, state_size, learning_rate=1e-3, attn_type='luong', arch=1):
        self.state_size = state_size
        self.lr = learning_rate
        self.attn_type = attn_type      # attention type
        self.arch = arch                # attention architecture
        # create NN models
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        img_input = layers.Input(shape=self.state_size)

        # first convolutional layers
        x = layers.Conv2D(16, kernel_size=5, strides=2,
                          padding="SAME", activation="relu")(img_input)

        if self.attn_type == 'bahdanau':
            attn = layers.AdditiveAttention()([x, x])      # Bahdanau-style
        elif self.attn_type == 'luong':
            attn = layers.Attention()([x, x])             # Luong-style
        else:
            raise ValueError("Choose between 'bahdanau' and 'loung' for attention type.")

        if self.arch == 1:
            x = attn
        elif self.arch == 2:
            x = layers.Add()([attn, x])
            #x = tf.keras.activations.sigmoid(x)
        elif self.arch == 3:
            x = layers.Multiply()([attn, x])
            #x = tf.keras.activations.sigmoid(x)
        else:
            raise ValueError('Choose 1, 2 or 3 for architecture')

        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # second conv layer
        x = layers.Conv2D(32, kernel_size=5, strides=2,
                          padding="SAME", activation="relu")(x)

        if self.attn_type == 'bahdanau':
            attn = layers.AdditiveAttention()([x, x])  # Bahdanau-style
        elif self.attn_type == 'luong':
            attn = layers.Attention()([x, x])  # Luong-style
        else:
            raise ValueError("Choose between 'bahdanau' and 'loung' for attention type.")

        if self.arch == 1:
            x = attn
        elif self.arch == 2:
            x = layers.Add()([attn, x])
            #x = tf.keras.activations.sigmoid(x)
        elif self.arch == 3:
            x = layers.Multiply()([attn, x])
            #x = tf.keras.activations.sigmoid(x)
        else:
            raise ValueError('Choose 1, 2 or 3 for architecture')

        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Third conv layer
        x = layers.Conv2D(64, kernel_size=5, strides=2,
                          padding="SAME", activation="relu")(x)

        if self.attn_type == 'bahdanau':
            attn = layers.AdditiveAttention()([x, x])  # Bahdanau-style
        elif self.attn_type == 'luong':
            attn = layers.Attention()([x, x])  # Luong-style
        else:
            raise ValueError("Choose between 'bahdanau' and 'loung' for attention type.")

        if self.arch == 1:
            x = attn
        elif self.arch == 2:
            x = layers.Add()([attn, x])
            #x = tf.keras.activations.sigmoid(x)
        elif self.arch == 3:
            x = layers.Multiply()([attn, x])
            #x = tf.keras.activations.sigmoid(x)
        else:
            raise ValueError('Choose 1, 2 or 3 for architecture')

        x = layers.LayerNormalization(epsilon=1e-6)(x)
        #x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        model = tf.keras.Model(inputs=img_input, outputs=x, name='feature_net')
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='att_feature_net.pdf',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)

