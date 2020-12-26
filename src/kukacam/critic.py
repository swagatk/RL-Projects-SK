"""
CRITIC MODEL
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class KukaCritic:
    def __init__(self, state_size, action_size,
                 replacement,
                 learning_rate,
                 gamma, feature_model):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.replacement = replacement
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.feature_model = feature_model
        self.model = self._build_net()
        self.target = self._build_net()
        self.gamma = gamma
        self.train_step_count = 0

    def _build_net(self):
        # state input is a stack of 1-D YUV images
        state_input = layers.Input(shape=self.state_size)

        feature = self.feature_model(state_input)
        state_out = layers.Dense(32, activation="relu")(feature)

        # Action as input
        action_input = layers.Input(shape=self.action_size)
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)
        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=net_out)
        model.summary()
        keras.utils.plot_model(model, to_file='critic_net.png',
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

    def train(self, state_batch, action_batch, reward_batch,
              next_state_batch, done_batch, actor): #
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            target_actions = actor.target(next_state_batch)
            target_critic = self.target([next_state_batch, target_actions])
            y = reward_batch + self.gamma * (1 - done_batch) * target_critic
            critic_value = self.model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


