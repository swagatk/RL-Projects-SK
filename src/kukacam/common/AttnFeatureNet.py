'''
Network for extracting features from input images
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common.SeqSelfAttention import SeqSelfAttention
###########################
# feature network
########################
class AttentionFeatureNetwork:
    def __init__(self, state_size, learning_rate=1e-3,
                 attn_type='seqself'):
        self.state_size = state_size
        self.lr = learning_rate
        self.attn_type = attn_type
        # create NN models
        self.model = self._build_net2()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        img_input = layers.Input(shape=self.state_size)

        # shared convolutional layers
        x = layers.Conv2D(16, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(img_input)

        #attn1 = layers.AdditiveAttention()([x, x])      # Bahdanau-style
        x = layers.Attention()([x, x])             # Luong-style
        # x = layers.AdditiveAttention()([x, x])
        #attn1 = tf.keras.activations.sigmoid(attn1)
        #x = layers.Add()([attn1, x])
        # x = layers.Multiply()([attn1, x])
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(x)

        x = layers.Attention()([x, x])
        #x = layers.AdditiveAttention()([x, x])     # Bahdanau-style
        #attn2 = layers.AdditiveAttention()([x, x])
        #attn2 = layers.Attention()([x, x])         # Luong-style
        #attn2 = tf.keras.activations.sigmoid(attn2)
        #x = layers.Add()([attn2, x])
        #x = layers.Multiply()([attn2, x])
        # x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(x)
        # bn3 = layers.BatchNormalization()(conv3)
        # include attention layer
        #x = layers.MultiHeadAttention(num_heads=2, key_dim=36)(x, x)        # does not work well
        #attn3 = layers.Attention()([x, x])
        x = layers.Attention()([x, x])
        # x = layers.AdditiveAttention()([x, x])
        #attn3 = layers.AdditiveAttention()([x, x])
        #attn3 = tf.keras.activations.sigmoid(attn3)
        #x = layers.Multiply()([attn3, x])
        #x = layers.Add()([attn3, x])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
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

    def _build_net2(self):
        img_input = layers.Input(shape=self.state_size)

        x = layers.Conv2D(16, kernel_size=5, strides=2,
                          padding="SAME", activation="relu")(img_input)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(32, kernel_size=5, strides=2,
                          padding="SAME", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(32, kernel_size=5, strides=2,
                          padding="SAME", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, kernel_size=5, strides=2,
                          padding="SAME", activation="relu")(x)

        if self.attn_type == 'luong':
            x = layers.Attention()([x, x])
        elif self.attn_type == 'bahdanau':
            x = layers.AdditiveAttention()([x, x])
        elif self.attn_type == 'seqself':
            x = layers.Flatten()(x)
            x = layers.Reshape((4, 16))(x)
            x = SeqSelfAttention(units=16)(x)
            x = layers.Reshape((1, 256))(x)

        x = layers.LayerNormalization(epsilon=1e-6)(x)

        f = layers.Flatten()(x)

        f = layers.Dense(128, activation="relu")(f)
        f = layers.Dense(128, activation="relu")(f)
        f = layers.Dense(64, activation="relu")(f)
        model = tf.keras.Model(inputs=img_input, outputs=f, name='feature_net')
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net.png',
                               show_shapes=True, show_layer_names=True)
        return model


    def __call__(self, state):
        return self.model(state)

