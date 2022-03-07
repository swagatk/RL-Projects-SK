'''
CURL implementation in Tensorflow 2.0
Source for Pytorch implementation: 
URL: https://github.com/MishaLaskin/curl
'''
import numpy as np
import tensorflow as tf

from common.siamese_network import Encoder, SiameseNetwork

class curlActor():
    """ MLP Actor Network"""
    def __init__(self, obs_shape, action_shape, feature_dim, hidden_dim,
                    action_upper_bound, log_std_min=-20, log_std_max=2, 
                    encoder_model=None):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_upper_bound = action_upper_bound

        self.encoder = encoder_model
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.obs_shape)
        if self.encoder is None:
            x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(inputs)
        else:
            x = self.encoder(inputs)
        x = tf.keras.layers.Dense(self.hidden_dim*2, activation='relu')(x)
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x)
        mu = tf.keras.layers.Dense(self.action_shape[0], activation='tanh')(x)
        mu = mu * self.action_upper_bound
        log_sig = tf.keras.layers.Dense(self.action_shape[0])(x)
        model = tf.keras.Model(inputs=inputs, outputs=[mu, log_sig])
        return model 

    
    def __call__(self, obs):
        # input is a tensor
        mu, log_sigma = self.model(obs)
        std = tf.math.exp(log_sigma)
        return mu, std


class curlCritic():
    def __init__(self, obs_shape, action_shape, hidden_dim, gamma, feature_model=None) -> None:
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.encoder = feature_model
        #self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.Q1 = self.build_model()
        self.Q2 = self.build_model()


    def build_model(self):
        obs = tf.keras.layers.Input(shape=self.obs_shape)
        action = tf.keras.layers.Input(shape=self.action_shape)  
        if self.encoder is None:
            x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(obs)
        else:
            x = self.encoder(obs)

        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x)
        xa = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(action)
        x = tf.keras.layers.Concatenate()([x, action])

        x = tf.keras.layers.Dense(self.hidden_dim*2, activation='relu')(x)
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu')(x)
        v = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=[obs, action], outputs=v)
        return model

    def __call__(self, state, action):
        # input: tensors
        q1_value = self.Q1([state, action])
        q2_value = self.Q2([state, action])
        return q1_value, q2_value

    def set_weights(self, q1_wts, q2_wts):
        self.Q1.set_weights(q1_wts)
        self.Q2.set_weights(q2_wts)

    def get_weights(self):
        q1_wts = self.Q1.get_weights()
        q2_wts = self.Q2.get_weights()
        return q1_wts, q2_wts

class CURL():
    """
    CURL
    - It is essentially a siamese twin network
    """
    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target) -> None:
        self.batch_size = batch_size
        self.encoder = critic.encoder       # query
        self.encoder_target = critic_target.encoder # key

        # bilinear product
        self.W = tf.Variable(tf.random.uniform(shape=(z_dim, z_dim)))

        # how to use this?
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def encode(self, x, ema=False):
        if ema:
            z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Use logits trick for CURL
        - Compute (B, B) matrix z_a * W * z_pos.T
        - positives are all diagonal elements
        - negatives are all other elements.
        - to compute loss use multi-class crossentropy with identity matrix for labels
        """

        wz = tf.matmul(self.W, z_pos.T) # (z_dim, B)
        logits = tf.matmul(z_a, wz) # (B, B)
        logits = logits - tf.math.reduce_max(logits, axis=1)[0][:, None]  # what is this?

        return logits 


class curlSacAgent():
    """ Curl Representation learning with SAC """
    def __init__(self, obs_shape, action_shape, 
                    hidden_dim=256, 
                    discount=0.99,
                    init_temperature=0.01,  # ??
                    alpha_lr=1e-3,          # ??
                    alpha_beta=0.9,         # ??
                    actor_lr=1e-3,
                    actor_beta=0.9,
                    actor_log_std_min=-10,
                    actor_log_std_max=2,
                    actor_update_freq=2,
                    critic_lr=1e-3,
                    critic_tau=0.005,
                    critic_target_update_freq=2,
                    encoder_feature_dim=50,
                    encoder_lr=1e-3,
                    encoder_tau=0.005,
                    num_layers=4,
                    num_filters=32,
                    cpc_update_freq=1,
                    log_interval=100,
                    curl_latent_dim=128
    ):
        self.discount = discount
        self.critic_tau = critic_tau 
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape
        self.curl_latent_dim = curl_latent_dim
        self.action_shape = action_shape

        self.actor = curlActor(
            obs_shape, action_shape, hidden_dim, encoder_feature_dim, 
            actor_log_std_min, actor_log_std_max, num_layers, num_filters
            )

        self.critic = curlCritic(
            obs_shape, action_shape, hidden_dim, encoder_feature_dim,
            num_layers, num_filters
        )
        
        self.critic_target = curlCritic(
            obs_shape, action_shape, hidden_dim, encoder_feature_dim,
            num_layers, num_filters
        )

        self.curl = CURL(obs_shape, encoder_feature_dim, self.curl_latent_dim,
                        self.critic, self.critic_target)

        # copy the weights from the main critic to target
        self.critic_target.model.set_weights(self.critic.model.get_weights())

        # what is this?
        self.log_alpha = tf.Variable(np.log(init_temperature), dtype=tf.float32)
        self.log_alpha.trainable = True
        self.log_alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr, beta_1=alpha_beta)
        
        # Check its use
        self.target_entropy = -np.prod(action_shape)

    def train(self):
        self.actor.train()
        self.critic.train()
        self.curl.train()

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def select_action(self, obs):  
        # obs: (H, W, C)
        assert obs.shape() == self.image_size, "observation shape is not correct"

        tf_state = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float32), 0)
        action, _ = self.actor(tf_state)
        action = action.numpy()[0]
        return action

    def sample_action(self, obs):
        pass

    def update_critic(self, obs, action, next_obs, reward, done):
        # obs: (B, H, W, C)
        # action: (B, A)
        # next_obs: (B, H, W, C)
        # reward: (B)
        # done: (B)

        _, policy_action, log_pi, _ = self.actor(next_obs)
        target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
        target_V = tf.minimum(target_Q1, target_Q2) - self.alpha * log_pi

        # update the critic
        with tf.GradientTape() as tape:
            # compute the value of the next state
            next_value = self.critic_target(next_obs)
            # compute the value of the current state
            value = self.critic(obs, action)
            # compute the target value
            target_value = reward + self.discount * next_value * (1-done)
            # compute the loss
            loss = tf.reduce_mean(tf.square(target_value - value))
        grads = tape.gradient(loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.model.trainable_variables))

    


        
        
