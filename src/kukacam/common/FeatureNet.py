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
        #self.model = self._build_net2() # bigger network
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self, conv_layers=[16, 32, 32], dense_layers=[128, 128, 64]):
        img_input = tf.keras.layers.Input(shape=self.state_size)

        loop_input = img_input
        for i in range(len(conv_layers)):
            conv = tf.keras.layers.Conv2D(conv_layers[i], kernel_size=5, strides=2,
                                padding="SAME", activation="relu")(loop_input) 
            bn = tf.keras.layers.BatchNormalization()(conv)
            loop_input = bn

        f = tf.keras.layers.Flatten()(bn)

        for i in range(len(dense_layers)):
            f = tf.keras.layers.Dense(dense_layers[i], activation="relu")(f)

        model = tf.keras.Model(inputs=img_input, outputs=f, name='feature_net')
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net_1.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def _build_net2(self, conv_layers=[16, 32, 32], dense_layers=[128, 128, 64]):
        img_input = layers.Input(shape=self.state_size)

        loop_input = img_input
        for i in range(len(conv_layers)):
            conv = tf.keras.layers.Conv2D(conv_layers[i], kernel_size=5,
                        strides=2, padding="SAME", activation="relu")(loop_input)
            pool = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, 
                        padding="SAME")(conv)
            loop_input = pool
            
        f = layers.Flatten()(pool)

        for i in range(len(dense_layers)):
            f = tf.keras.layers.Dense(dense_layers[i], activation="relu")(f)

        model = tf.keras.Model(inputs=img_input, outputs=f, name='feature_net_2')
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net_2.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)


##################################
'''
attention layer is applied in between conv layers. Typical architecture looks like:
Conv2d - Attention - Conv2d - Attention - Conv2D - Attention - Flatten - Dense Layers

attn_type: 'bahdanau', 'luong'
    - 'bahdanau': keras.layers.AdditiveAttention()
    - 'luong': keras.layers.Attention()

Architectures:
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

    def _build_net(self, conv_layers=[16, 32, 32], dense_layers=[128, 128, 64]):
        img_input = layers.Input(shape=self.state_size)

        loop_input = img_input
        for i in range(len(conv_layers)):
            # Conv Layers
            x = tf.keras.layers.Conv2D(conv_layers[i], kernel_size=5, strides=2,
                                padding="SAME", activation="relu")(loop_input)
            # Attention
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

            x = layers.LayerNormalization(epsilon=1e-6)(x)
            loop_input = x

        x = layers.Flatten()(x)

        # Fully-Connected layers
        for i in range(len(dense_layers)):
            x = tf.keras.layers.Dense(dense_layers[i], activation="relu")(x)
        
        model = tf.keras.Model(inputs=img_input, outputs=x, name='attn_feature_net')
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='att_feature_net.pdf',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)

