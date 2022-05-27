from unicodedata import name
from cv2 import CirclesGridFinderParameters
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
from encoder import Encoder
from src.common.utils import random_crop

sys.path.append('/home/swagat/GIT/RL-Projects-SK/src/common/')
sys.path.append('/home/swagat/GIT/RL-Projects-SK/src/algo/')

from common.buffer import Buffer


### Actor Network
class CurlActor:
    def __init__(self, state_size, action_size,
        action_upper_bound,
        encoder_feature_dim,
        learning_rate=1e-3,
        actor_dense_layers=[128, 64],
        save_model_plot=False) -> None:
        self.state_size = state_size # shape: (h, w, c)
        self.action_size = action_size
        self.lr = learning_rate
        self.action_uppper_bound = action_upper_bound
        self.encoder_feature_dim = encoder_feature_dim
        self.actor_dense_layers = actor_dense_layers
        self.save_model_plot = save_model_plot

        self.encoder = Encoder(obs_shape=self.state_size,
                                feature_dim=self.encoder_feature_dim)


        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):

        inp = tf.keras.layers.Input(shape=self.state_size)
        x = inp
        for i in range(len(self.actor_dense_layers)):
            x = tf.keras.layers.Dense(self.actor_dense_layers[i],
                        activation='relu')(x)
        mu = tf.keras.layers.Dense(self.action_size[0],
                            activation='tanh')(x)
        log_sig = tf.keras.layers.Dense(self.action_size[0])(x)
        mu = mu * self.action_uppper_bound
        model = tf.keras.Model(inputs=inp, outputs=[mu, log_sig],
                            name='actor')
        model.summary()
        if self.save_model_plot:
            tf.keras.utils.plot_model(model,
                                to_file='actor_network.png',
                                show_shapes=True,
                                show_layer_names=True)
        return model

    def  __call__(self, state):
        mu, log_sig = self.model(state)
        std = tf.math.exp(log_sig)
        return mu, std 

    def policy(self, state):

        mean, std = self.__call__(state)

        # sample actions from normal distribution
        pi = tfp.distributions.Normal(mean, std)
        action_ = pi.sample()
        log_pi_ = pi.log_prob(action_)
        
        # Apply squashing function
        action = tf.clip_by_value(action_, 
                    -self.action_uppper_bound,
                    self.action_uppper_bound)
        
        log_pi_a = log_pi_ - tf.reduce_sum(
            tf.math.log(tf.keras.activations.relu(1 - action ** 2) + 1e-6)
                            axis=-1, keepdims=True)
        return action, log_pi_a 


    def train(self):
        pass

    def save_weights(self, filename):
        self.model.save_weights(filename, save_format='h5')

    def load_weights(self, filename):
        self.model.load_weights(filename)

###############33

        
#########3
class CurlCritic:
    def __init__(self, state_size, action_size,
                encoder_feature_dim,
                learning_rate = 1e-3,
                gamma=0.95,
                critic_dense_layers=[32, 32],
                save_model_plot=False,
                encoder=None) -> None:

        self.state_size = state_size
        self.action_size = action_size
        self.encoder_feature_dim = encoder_feature_dim
        self.lr = learning_rate
        self.gamma = gamma 
        self.critic_dense_layers=critic_dense_layers
        self.save_model_plot = save_model_plot

        if encoder is None:
            self.encoder = Encoder(
                            obs_shape=self.state_size,
                            feature_dim=self.encoder_feature_dim
                            )
        else:
            self.encoder = encoder
            self.encoder_feature_dim = self.encoder.model.outputs[0].shape[-1]

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_model(self):
        state_input = tf.keras.layers.Input(shape=self.state_size)
        features = self.encoder(state_input)
        action_input = tf.keras.layers.Input(shape=self.action_size)

        combined_input = tf.keras.layers.Concatenate()([features, action_input])

        x = combined_input
        for i in range(len(self.critic_dense_layers)):
            x = tf.keras.layers.Dense(self.critic_dense_layers[i],
                                    activation='relu')(x)
        net_out = tf.keras.layers.Dense(1)(x)

        self.model = tf.keras.Model(inputs=[state_input, action_input], 
                                    outputs=net_out,
                                    name='critic')
        self.model.summary()

        if self.save_model_plot:
            tf.keras.utils.plot_model(self.model,
                                to_file='critic_network.png',
                                show_shapes=True,
                                show_layer_names=True)


    def __call__(self, state, action):
        q_value = self.model([state, action])
        return q_value

    def train(self):
        pass

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


#####################
class CURL:
    def __init__(self, obs_shape, z_dim, batch_size,
                critic, target_critic,
                learning_rate=1e-3) -> None:
        
        self.batch_size = batch_size
        self.lr = learning_rate
        self.z_dim = z_dim      # feature dimension
        self.encoder = critic.encoder
        self.encoder_target = target_critic.encoder 
        self.W = tf.Variable(shape=(self.z_dim, self.z_dim))
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def encode(self, x, ema=False):
        if ema:
            z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        return z_out 

    def compute_logits(self, z_a, z_pos):
        Wz = tf.matmul(self.W, tf.transpose(z_pos)) # (z_dim, B)
        logits = tf.matmul(z_a, Wz)     # (B, B)
        logits = logits - tf.reduce_max(logits, axis=1)
        labels = tf.range(logits.shape[0])
        return logits, labels 

    def train(self, x_a, x_pos):
        """ train the model on the data
        data: tuple (obs & augmented obs): obs tensor of shape: (batch_size, height, width, channels) 
        """
        with tf.GradientTape() as tape1:
            z_a = self.encode(x_a)
            z_pos = self.encode(x_pos, ema=True)
            logits, labels = self.compute_logits(z_a, z_pos)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, axis=-1)
            train_wts = [self.W]
        gradients_1 = tape1.gradient(loss, train_wts)
        self.optimizer.apply_gradients(zip(gradients_1, train_wts))

        with tf.GradientTape() as tape2:
            z_a = self.encode(x_a)
            z_pos = self.encode(x_pos, ema=True)
            logits, labels = self.compute_logits(z_a, z_pos)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            enc_wts = self.encoder.model.trainable_variables
        gradients_2 = tape2.gradient(loss, enc_wts)
        self.encoder.optimizer.apply_gradients(zip(gradients_2, enc_wts))
        return loss 

    
##############
class CurlSacAgent:
    def __init__(self, state_size, action_size, action_upper_bound,
                buffer_capacity=100000,
                batch_size=128,
                lr=1e-3,        # learning rate
                alpha=0.2,      # entropy coefficient
                gamma=0.99,     # discount factor
                polyak=0.995,   # soft update factor: tau
                curl_feature_dim=128,    # latent feature dim
                cropped_img_size=84,
                stack_size=3,   # not used
                init_steps=500,
                eval_freq=1000,
                critic_target_update_freq=2,
                actor_target_update_freq=2,
                encoder_target_update_freq=2,

                ):
            
            self.state_size = state_size
            self.action_shape = action_size
            self.action_upper_bound = action_upper_bound
            self.lr = lr 
            self.buffer_capacity = buffer_capacity
            self.batch_size = batch_size
            self.gamma = gamma 
            self.polyak = polyak 
            self.feature_dim = curl_feature_dim
            self.cropped_img_size = cropped_img_size
            self.stack_size = stack_size # not used
            
            assert self.state_size.ndim == 3, 'image observation of shape (h, w, c)'

            self.obs_shape = (self.cropped_img_size, self.cropped_img_size, self.state_size[2])

            # Actor
            self.actor = CurlActor(
                    state_size=self.obs_shape, 
                    action_size=self.action_shape,
                    action_upper_bound=self.action_upper_bound,
                    encoder_feature_dim=self.feature_dim
                    )

            # Create two Critic
            self.critic_1 = CurlCritic(
                    state_size=self.obs_shape,
                    action_size=self.action_shape,
                    encoder_feature_dim=self.feature_dim,
                    learning_rate=self.lr
                    )

            self.critic_2 = CurlCritic(
                    state_size=self.obs_shape,
                    action_size=self.action_shape,
                    encoder_feature_dim=self.feature_dim,
                    learning_rate=self.lr,
                    encoder=self.critic_1.encoder, 
                    )

            # create two target critics
            self.target_critic_1 = CurlCritic(
                state_size=self.obs_shape,
                action_size=self.action_shape,
                encoder_feature_dim=self.feature_dim,
                learning_rate=self.lr
                )

            self.target_critic_2 = CurlCritic(
                state_size=self.obs_shape,
                action_size=self.action_shape,
                encoder_feature_dim=self.feature_dim,
                learning_rate=self.lr,
                encoder=self.target_critic_1.encoder,
                )

            # critic & target critic share weights initially
            self.target_critic_1.model.set_weights(self.critic_1.model.get_weights())
            self.target_critic_2.model.set_weights(self.critic_2.model.get_weights())

            # CURL agent
            self.curl = CURL(
                obs_shape=self.obs_shape,
                z_dim = self.feature_dim,
                batch_size=self.batch_size,
                critic=self.critic_1,
                target_critic=self.target_critic_1
            )

            # entropy coefficient as a tunable parameter
            self.alpha = tf.Variable(alpha, dtype=tf.float32)
            self.alpha_optimizer = tf.keras.optimizers.Adam(self.lr)
            self.target_entropy = - tf.constant(np.prod(self.action_shape), dtype=tf.float32)

            # Buffer
            self.buffer = Buffer(self.buffer_capacity,
                                    self.batch_size)

    def sample_action(self, state):
        # input: numpy array
        # output: numpy array
        if state.ndim < len(self.obs_shape) + 1:      # single sample
            tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        else:
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)

        action, log_pi = self.actor.policy(tf_state)   # returns tensors
        return action[0].numpy(), log_pi[0].numpy()

    def update_alpha(self, states):
        # input: tensor
        with tf.GradientTape() as tape:
            # sample actions from the policy for the current states
            _, log_pi_a = self.actor.policy(states)
            alpha_loss = tf.reduce_mean(-self.alpha * (log_pi_a + self.target_entropy))
            variables = [self.alpha]
        # outside gradient tape block
        grads = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))
        return alpha_loss.numpy()

    def update_critic_networks(self, states, actions, rewards, next_states, dones):

        with tf.GradientTape() as tape1: 
            q1 = self.critic_1(states, actions)
            pi_a, log_pi_a = self.actor.policy(next_states)
            q1_target = self.target_critic_1(next_states, pi_a)
            q2_target = self.target_critic_2(next_states, pi_a)
            min_q_target = tf.minimum(q1_target, q2_target)
            soft_q_target = min_q_target - self.alpha * log_pi_a 
            y = tf.stop_gradient(rewards + self.gamma * soft_q_target * (1 - dones))
            critic_1_loss = tf.reduce_mean(tf.square(y - q1))
        grads1 = tape1.gradient(critic_1_loss, self.critic_1.model.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(grads1, self.critic_1.model.trainable_variables))

        with tf.GradientTape() as tape2:
            q2 = self.critic_2(states, actions)
            pi_a, log_pi_a = self.actor.policy(next_states)
            q1_target = self.target_critic_1(next_states, pi_a)
            q2_target = self.target_critic_2(next_states, pi_a)
            min_q_target = tf.minimum(q1_target, q2_target)
            soft_q_target = min_q_target - self.alpha * log_pi_a 
            y = tf.stop_gradient(rewards + self.gamma * soft_q_target * (1 - dones))
            critic_2_loss = tf.reduce_mean(tf.square(y - q2))
        grads2 = tape2.gradient(critic_2_loss, self.critic_2.model.trainable_variables)
        self.critic_2.optimizer.apply_gradients(zip(grads2, self.critic_2.model.trainable_variables))
        return tf.minimum(critic_1_loss, critic_2_loss).numpy()

    def update_actor_network(self, states):
        with tf.GradientTape() as tape:
            pi_a, log_pi_a = self.actor.policy(states)
            q1 = self.critic_1(states, pi_a)
            q2 = self.critic_2(states, pi_a)
            min_q = tf.minimum(q1, q2)
            soft_q = min_q - self.alpha * log_pi_a
            actor_loss = tf.reduce_mean(soft_q)
        grads = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.model.trainable_variables))
        return actor_loss.numpy()

    def update_target_networks(self):
        for wt_target, wt_source in zip(self.target_critic_1.model.trainable_variables, 
                                self.critic_1.model.trainable_variables):
            wt_target = self.polyak * wt_target + (1 - self.polyak) * wt_source

        for wt_target, wt_source in zip(self.target_critic_2.model.trainable_variables, 
                            self.critic_2.model.trainable_variables):
            wt_target = self.polyak * wt_target + (1 - self.polyak) * wt_source

        for wt_target, wt_source in zip(self.target_encoder.model.trainable_variables,
                                    self.encoder.model.trainable_variable):
            wt_target = self.polyak * wt_target + (1 - self.polyak) * wt_source

    def train(self, step):
        # sample a batch of data
        states, actions, rewards, next_states, dones = self.buffer.sample()

        states = random_crop(states, self.cropped_img_size)
        next_states = random_crop(next_states, self.cropped_img_size)


        # update the critic
        critic_loss = self.critic.train(states, actions, rewards, next_states, dones)
        # update the actor
        actor_loss = self.curl.train(states, actions)

        return critic_loss, actor_loss








            




    

