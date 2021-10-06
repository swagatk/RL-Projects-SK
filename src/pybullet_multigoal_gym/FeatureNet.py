'''
Network for extracting features from input images

- It is now possible to visualize attention scores 
'''
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
#sys.path.append(os.path.dirname(current_dir))

###########################
# CNN feature network
########################
class FeatureNetwork:
    def __init__(self, state_size, attn: dict = None, learning_rate=1e-3):
        self.state_size = state_size
        self.lr = learning_rate
        self.attn = attn
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        self.model = self._build_net()
    
    def _build_net_2(self, conv_layers=[16, 32, 32], dense_layers=[128, 128, 64]):
        # One attention layer after Conv layers
        img_input = tf.keras.layers.Input(shape=self.state_size)

        x = img_input
        for i in range(len(conv_layers)):
            x = tf.keras.layers.Conv2D(conv_layers[i], kernel_size=5, strides=2,
                                padding="SAME", activation="relu")(x) 

            # Apply Batch normalization
            x = tf.keras.layers.BatchNormalization()(x)

        # outside for loop

        if self.attn is not None: 
            # Attention types
            if self.attn['type'] == 'luong':
                attn = tf.keras.layers.Attention()([x, x])
            elif self.attn['type'] == 'bahdanau':
                attn = tf.keras.layers.AdditiveAttention()([x, x])
            else:
                raise ValueError('Wrong type of attention. Exiting ...')

            # Attention architectures 
            if self.attn['arch'] == 0: 
                x = attn
            elif self.attn['arch'] == 1: 
                x = tf.keras.layers.Add()([attn, x])
            elif self.attn['arch'] == 2:
                x = tf.keras.layers.Multiply()([attn, x])
            elif self.attn['arch'] == 3:
                x = tf.keras.layers.Multiply()([attn, x])
                x = tf.keras.activations.sigmoid(x)
            else:
                raise ValueError('Wrong choice for attention architecture. Exiting ..')

        # Flatten the output
        x = tf.keras.layers.Flatten()(x)

        # Apply Dense layers
        for i in range(len(dense_layers)):
            x = tf.keras.layers.Dense(dense_layers[i], activation="relu")(x)

        model = tf.keras.Model(inputs=img_input, outputs=x, name='feature_net')
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net_1.png',
                               show_shapes=True, show_layer_names=True)
        return model
    
    def _build_net(self, conv_layers=[16, 32, 64, 64], dense_layers=[128, 128, 64, 32]):
        img_input = tf.keras.layers.Input(shape=self.state_size)

        x = img_input
        # Conv layers
        for i in range(len(conv_layers)):
            x = tf.keras.layers.Conv2D(conv_layers[i], kernel_size=5, strides=2,
                                padding="SAME", activation="relu")(x) 

            
            if self.attn is not None: 
                # Attention types
                if self.attn['type'] == 'luong':
                    if self.attn['return_scores']:
                        attn, scores = tf.keras.layers.Attention()([x, x], 
                                return_attention_scores=self.attn['return_scores'])
                    else:
                        attn = tf.keras.layers.Attention()([x, x])
                elif self.attn['type'] == 'bahdanau':
                    if self.attn['return_scores']:
                        attn, scores = tf.keras.layers.AdditiveAttention()([x, x], 
                                return_attention_scores=self.attn['return_scores'])
                    else:
                        attn = tf.keras.layers.AdditiveAttention()([x, x])
                else:
                    raise ValueError('Wrong type of attention. Exiting ...')

                # Attention architectures 
                if self.attn['arch'] == 0: 
                    x = attn
                elif self.attn['arch'] == 1: 
                    x = tf.keras.layers.Add()([attn, x])
                elif self.attn['arch'] == 2:
                    x = tf.keras.layers.Multiply()([attn, x])
                elif self.attn['arch'] == 3:
                    x = tf.keras.layers.Multiply()([attn, x])
                    x = tf.keras.activations.sigmoid(x)
                else:
                    raise ValueError('Wrong choice for attention architecture. Exiting ..')
            
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, 
                        padding="SAME")(x)
        # end of for-loop

        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Flatten the output
        x = tf.keras.layers.Flatten()(x)

        # Apply Dense layers
        for i in range(len(dense_layers)):
            x = tf.keras.layers.Dense(dense_layers[i], activation="relu")(x)

        if self.attn is not None and self.attn['return_scores'] is True:
            model = tf.keras.Model(inputs=img_input, outputs=[x, scores], name='feature_net')
        else:
            model = tf.keras.Model(inputs=img_input, outputs=x, name='feature_net')

        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net_1.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state): 
        if self.attn is not None and self.attn['return_scores'] is True: 
            feature, _ = self.model(state)
        else:
            feature = self.model(state)
        return feature 

    def get_attention_scores(self, state):
        if self.attn is not None and self.attn['return_scores'] is True:
            feature, scores = self.model(state)
            return feature.numpy(), scores.numpy()
        else:
            raise ValueError("Enable appropriate attention flags for this function." )
    


##################################
'''
attention layer is applied in between conv layers. Typical architecture looks like:
Conv2d - Attention - Conv2d - Attention - Conv2D - Attention - Flatten - Dense Layers

attn_type: 'bahdanau', 'luong'
    - 'bahdanau': keras.layers.AdditiveAttention()
    - 'luong': keras.layers.Attention()

Architectures:
arch = 0: attention(x) in-between conv layers
arch = 1: x + attention(x) in-between conv layers
arch = 2: x * attention(x) in-between conv layers
arch = 3: sigmoid ( x * attention (x)) between conv layers

x is the output from previous conv layer

This class is now merged with the FeatureNetwork class and may be removed.

'''

###################################
#   CNN + LSTM FEATURE NETWORK
##################################

class CNNLSTMFeatureNetwork:
    def __init__(self, state_size: tuple,  attn: dict, learning_rate=1e-3) -> None:
        self.state_size = state_size    # shape: 4 dim: stack_size, h, w, c
        self.lr = learning_rate
        self.step_counter = 0
        self.attn = attn
        self.stack_size = self.state_size[0]    # verify this ... 

        self.model = self._build_net()      # two different architectures for applying attentino
        # self.model = self._build_net_2()

        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    
    def _build_net(self, conv_layers=[16, 32, 32], 
                            dense_layers=[128, 128, 64]):

        # return attention scores
        if self.attn is not None and self.attn['return_scores'] is True:
            attn_scores = []

        org_input = tf.keras.layers.Input(shape=self.state_size)
        x = org_input 
        for i in range(len(conv_layers)):
            x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv2D(conv_layers[i], 
                            kernel_size=5, strides=2,
                            padding="SAME", activation="relu"))(x)

            if self.attn is not None: 
                if self.attn['type'] == 'luong':
                    if self.attn['return_scores']:
                        attn, scores = tf.keras.layers.Attention()([x, x], 
                            return_attention_scores=self.attn['return_scores'])
                    else:
                        attn = tf.keras.layers.Attention()([x, x])
                elif self.attn['type'] == 'bahdanau':
                    if self.attn['return_scores']:
                        attn, scores = tf.keras.layers.AdditiveAttention()([x, x], 
                            return_attention_scores=self.attn['return_scores'])
                    else:
                        attn = tf.keras.layers.AdditiveAttention()([x, x])
                else:
                    raise ValueError('Wrong type of attention. Exiting ...')
                
                # store attention scores for each layer
                if self.attn['return_scores']:
                    attn_scores.append(scores)
            
                # Attention architectures 
                if self.attn['arch'] == 0: 
                    x = attn
                elif self.attn['arch'] == 1: 
                    x = tf.keras.layers.Add()([attn, x])
                elif self.attn['arch'] == 2:
                    x = tf.keras.layers.Multiply()([attn, x])
                elif self.attn['arch'] == 3:
                    x = tf.keras.layers.Multiply()([attn, x])
                    x = tf.keras.activations.sigmoid(x)
                else:
                    raise ValueError('Wrong choice for attention architecture. Exiting ..')

            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                strides=None, padding="SAME"))(x)
        # end of for loop

        x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten())(x)

        x = tf.keras.layers.LSTM(self.stack_size, activation="relu", 
                            return_sequences=False)(x)

        # Dense layers
        for i in range(len(dense_layers)):
            x = tf.keras.layers.Dense(dense_layers[i], 
                                    activation="relu")(x)

        if self.attn is not None and self.attn['return_scores'] is True:
            model = tf.keras.Model(inputs=org_input, outputs=[x, attn_scores], name='cnn_lstm_feature_net')
        else:
            model = tf.keras.Model(inputs=org_input, outputs=x, name='cnn_lstm_feature_net')

        model.summary()
        keras.utils.plot_model(model, to_file='cnn_lstm_feature_net.png',
                        show_shapes=True, show_layer_names=True)
        return model 


    def _build_net_2(self, conv_layers=[16, 32, 32], 
                            dense_layers=[128, 128, 64]):
        # there is only one attention layer after flatten

        org_input = tf.keras.layers.Input(shape=self.state_size)
        x = org_input 
        for i in range(len(conv_layers)):
            x = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv2D(conv_layers[i], 
                            kernel_size=5, strides=2,
                            padding="SAME", activation="relu"))(x)

            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                strides=None, padding="SAME"))(x)
        # end of for loop

        x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten())(x)
                
        # apply attention layer
        if self.attn is not None: 
            if self.attn['type'] == 'luong':
                if self.attn['return_scores']:
                    attn, scores = tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Attention())([x, x], 
                        return_attention_scores=self.attn['return_scores'])
                else:
                    attn = tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Attention())([x, x])
            elif self.attn['type'] == 'bahdanau':
                if self.attn['return_scores']:
                    attn, scores = tf.keras.layers.TimeDistributed(
                        tf.keras.layers.AdditiveAttention())([x, x], 
                        return_attention_scores=self.attn['return_scores'])
                else:
                    attn = tf.keras.layers.TimeDistributed(
                        tf.keras.layers.AdditiveAttention())([x, x])
            else:
                raise ValueError('Wrong type of attention. Exiting ...')
            
        
            # Attention architectures 
            if self.attn['arch'] == 0: 
                x = attn
            elif self.attn['arch'] == 1: 
                x = tf.keras.layers.Add()([attn, x])
            elif self.attn['arch'] == 2:
                x = tf.keras.layers.Multiply()([attn, x])
            elif self.attn['arch'] == 3:
                x = tf.keras.layers.Multiply()([attn, x])
                x = tf.keras.activations.sigmoid(x)
            else:
                raise ValueError('Wrong choice for attention architecture. Exiting ..')

        # LSTM layer
        x = tf.keras.layers.LSTM(self.stack_size, activation="relu", 
                            return_sequences=False)(x)

        # Dense layers
        for i in range(len(dense_layers)):
            x = tf.keras.layers.Dense(dense_layers[i], 
                                    activation="relu")(x)

        if self.attn is not None and self.attn['return_scores'] is True:
            model = tf.keras.Model(inputs=org_input, outputs=[x, scores], name='cnn_lstm_feature_net')
        else:
            model = tf.keras.Model(inputs=org_input, outputs=x, name='cnn_lstm_feature_net')

        model.summary()
        keras.utils.plot_model(model, to_file='cnn_lstm_feature_net.png',
                        show_shapes=True, show_layer_names=True)
        return model 

    def __call__(self, state):
        # input is a tensor of shape (-1, h, w, c)
        if self.attn is not None and self.attn['return_scores'] is True:
            feature, _ = self.model(state)
        else:
            feature = self.model(state)
            return feature 

    def get_attention_scores(self, state):
        # input is a tensor
        if self.attn is not None and self.attn['return_scores'] is True:
            feature, scores = self.model(state)
            return feature, scores
        else:
            raise ValueError('Please Enable suitable Attention Flags')


