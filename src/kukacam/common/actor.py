"""
ACTOR MODEL
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class KukaActor:
    def __init__(self, state_size, action_size,
                 replacement, learning_rate,
                 upper_bound, feature_model):
        self.state_size = state_size   # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.replacement = replacement
        self.upper_bound = upper_bound
        self.train_step_count = 0

        # create NN models
        self.feature_model = feature_model
        self.model = self._build_net()
        self.target = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        # input is a stack of 1-channel YUV images
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

        state_input = layers.Input(shape=self.state_size)
        feature = self.feature_model(state_input)

        l2 = layers.Dense(64, activation="relu")(feature)
        net_out = layers.Dense(self.action_size[0], activation='tanh',
                               kernel_initializer=last_init)(l2)

        net_out = net_out * self.upper_bound  # element-wise product
        model = keras.Model(state_input, net_out)
        model.summary()
        keras.utils.plot_model(model, to_file='actor_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def update_target(self):

        if self.replacement['name'] == 'hard':
            if self.train_step_count % \
                    self.replacement['rep_iter_a'] == 0:
                self.target.set_weights(self.model.get_weights())
        else:
            w = np.array(self.model.get_weights())
            w_dash = np.array(self.target.get_weights())
            new_wts = self.replacement['tau'] * w + \
                      (1 - self.replacement['tau']) * w_dash
            self.target.set_weights(new_wts)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def train(self, state_batch, critic):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            actor_weights = self.model.trainable_variables
            actions = self.model(state_batch)
            critic_value = critic.model([state_batch, actions])
            # -ve value is used to maximize value function
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss


