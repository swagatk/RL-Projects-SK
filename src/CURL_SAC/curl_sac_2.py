"""
Main changes compared to curl_sac_2.py

- We are using a common encoder for all actor/critic networks


Updates:
27/07/2022: 
    - Modifying CurlCritic to return q1 & q2 together. 
    - This makes CurlSACAgent code more cleaner. 
    - Incorporating reconstruction loss
    - It should supercede `curl_sac.py` file.

31/07/2022:
    - Incorporates consistency Loss

01/08/2022:
    - Possible BUG: Having a common encoder is problematic. Every model is trying to
        update this encoder. If it is frozen inside a class, it becomes 
        frozen everywhere. Probably, it is a better idea to have different
        encoders for each model and share the weights betweeen them.  
    - Its not a bug anymore. There is no other way to share weights between
        actor & critic encoders. Note that the actor & critic encoders are getting
        updated indepedently by the SAC algorithm. It is important to tie the 
        encoders together. So copying at regular intervals is not a good idea as the
        encoder learn will get lost when the weights are copied. 
    - The target critic encoder should not be tied to critic encoder. They should
        be separate. This is a new change here. 
    
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import os
import wandb
import pickle

#sys.path.append('/home/swagat/GIT/RL-Projects-SK/src/common')
#sys.path.append('/home/swagat/GIT/RL-Projects-SK/src/algo')
#sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, '/home/swagat/GIT/RL-Projects-SK/src/algo/common')

from common.buffer import Buffer
from encoder import Decoder, Encoder, FeaturePredictor
from common.utils import uniquify
from augmentation import random_crop, center_crop_image


### Actor Network
class CurlActor:
    def __init__(self, state_size, action_size,
        action_upper_bound,
        encoder_feature_dim,
        learning_rate=1e-3,
        actor_dense_layers=[128, 64],
        save_model_plot=False,
        model_name='actor',
        encoder=None,
        frozen_encoder=False) -> None:

        self.state_size = state_size # shape: (h, w, c)
        self.action_size = action_size
        self.lr = learning_rate
        self.action_uppper_bound = action_upper_bound
        self.encoder_feature_dim = encoder_feature_dim
        self.actor_dense_layers = actor_dense_layers
        self.save_model_plot = save_model_plot
        self.frozen_encoder = frozen_encoder
        self.model_name = model_name

        if encoder is None:
            self.encoder = Encoder(obs_shape=self.state_size,
                                feature_dim=self.encoder_feature_dim)
        else:
            self.encoder = encoder
            self.encoder_feature_dim = self.encoder.model.outputs[0].shape[-1]


        self.model = self._build_net()

        if self.frozen_encoder: 
            # Freeze encoder weights during RL training
            self.model.get_layer('encoder').trainable=False 

        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        inp = tf.keras.layers.Input(shape=self.state_size)
        x = self.encoder(inp)
        for i in range(len(self.actor_dense_layers)):
            x = tf.keras.layers.Dense(self.actor_dense_layers[i],
                        activation='relu')(x)
        mu = tf.keras.layers.Dense(self.action_size[0],
                            activation='tanh')(x)
        log_sig = tf.keras.layers.Dense(self.action_size[0])(x)
        mu = mu * self.action_uppper_bound
        model = tf.keras.Model(inputs=inp, outputs=[mu, log_sig],
                            name=self.model_name)
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

        #mean, std = self.__call__(state)
        mean, std = self(state)

        # sample actions from normal distribution
        pi = tfp.distributions.Normal(mean, std)
        action_ = pi.sample()
        log_pi_ = pi.log_prob(action_)
        
        # Apply squashing function
        action = tf.clip_by_value(action_, 
                    -self.action_uppper_bound,
                    self.action_uppper_bound)
        
        log_pi_a = log_pi_ - tf.reduce_sum(
            tf.math.log(tf.keras.activations.relu(1 - action ** 2) + 1e-6),
                            axis=-1, keepdims=True)
        return action, log_pi_a 

    def train(self):
        pass

    def save_weights(self, filename):
        self.model.save_weights(filename, save_format='h5')

    def load_weights(self, filename):
        self.model.load_weights(filename)

###############33

class QFunction():
    """
    MLP for Q Function
    """
    def __init__(self, feature_dim, action_dim,
                    dense_layers=[32, 32],
                    model_name='q_function') -> None:
        self.input_dim = feature_dim + action_dim
        self.dense_layers = dense_layers
        self.model_name = model_name
        self.model = self._build_net()
        
    def _build_net(self):
        input = tf.keras.layers.Input(shape=(self.input_dim,))
        x = input
        for i in range(len(self.dense_layers)):
            x = tf.keras.layers.Dense(self.dense_layers[i],
                        activation='relu')(x)
        q = tf.keras.layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=input, outputs=q, name=self.model_name)
        model.summary()
        return model
    
    def __call__(self, x):
        q = self.model(x)
        return q

        
        
#########3
class CurlCritic:
    def __init__(self, state_size, action_size,
                encoder_feature_dim,
                learning_rate = 1e-3,
                gamma=0.95,
                critic_dense_layers=[32, 32],
                save_model_plot=False,
                model_name='critic',
                encoder=None,
                frozen_encoder=False) -> None:

        self.state_size = state_size
        self.action_size = action_size
        self.encoder_feature_dim = encoder_feature_dim
        self.lr = learning_rate
        self.gamma = gamma 
        self.critic_dense_layers=critic_dense_layers
        self.save_model_plot = save_model_plot 
        self.frozen_encoder = frozen_encoder
        self.model_name = model_name

        self.Q1 = QFunction(self.encoder_feature_dim, 
                        self.action_size[0], model_name='q1_function')
        self.Q2 = QFunction(self.encoder_feature_dim, 
                        self.action_size[0], model_name='q2_function')


        if encoder is None:
            self.encoder = Encoder(
                            obs_shape=self.state_size,
                            feature_dim=self.encoder_feature_dim
                            )
        else:
            self.encoder = encoder
            self.encoder_feature_dim = self.encoder.model.outputs[0].shape[-1]

        
        self.model = self._build_model()

        if self.frozen_encoder:
            # Freeze encoder weights during RL training
            self.model.get_layer('encoder').trainable=False 

        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_model(self):
        state_input = tf.keras.layers.Input(shape=self.state_size)
        state_feats = self.encoder(state_input)
        action_input = tf.keras.layers.Input(shape=self.action_size)
        x = tf.keras.layers.Concatenate()([state_feats, action_input])

        q1 = self.Q1(x)
        q2 = self.Q2(x)

        model = tf.keras.Model(inputs=[state_input, action_input], 
                                    outputs=[q1, q2],
                                    name=self.model_name)
        if self.save_model_plot:
            tf.keras.utils.plot_model(self.model,
                                to_file='critic_network.png',
                                show_shapes=True,
                                show_layer_names=True)
        return model

    def __call__(self, state, action):
        q1, q2 = self.model([state, action])
        return q1, q2

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
                learning_rate=1e-3,
                include_reconst_loss=False,
                include_consistency_loss=False) -> None:
        
        self.obs_shape = obs_shape
        self.batch_size = batch_size
        self.lr = learning_rate
        self.z_dim = z_dim      # feature dimension
        self.encoder = critic.encoder
        self.encoder_target = target_critic.encoder 
        self.include_reconst_loss = include_reconst_loss
        self.include_consistency_loss = include_consistency_loss

        if self.z_dim != self.encoder.feature_dim:
            self.z_dim = self.encoder.feature_dim

        # Required for finding cosine similarity = X^T * W * X
        self.W = tf.Variable(tf.random.uniform(shape=(self.z_dim, self.z_dim),
                                                minval=-0.1, maxval=0.1))
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # Required for including reconstruction loss
        if self.include_reconst_loss:
            self.decoder = Decoder(obs_shape=self.obs_shape,
                                feature_dim=self.z_dim)

        # Required for including consistency loss
        if self.include_consistency_loss:
            self.feature_predictor = FeaturePredictor(self.z_dim)

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

    def train(self, aug_obs_anchor, aug_obs_pos, obs):
        """
        Args:
            aug_obs_anchor: (B, H, W, C) - anchor observations
            aug_obs_pos: (B, H, W, C) - positive observations
            obs: (B, H, W, C) - original observations before augmentation
        """

        # train encoder 
        curl_loss = self.train_encoder(aug_obs_anchor, aug_obs_pos, obs)

        # train decoder
        if self.include_reconst_loss:
            decoder_loss = self.train_decoder(obs)

        # train feature predictor
        if self.include_consistency_loss:
            predictor_loss = self.train_feature_predictor(obs, aug_obs_anchor)

        return curl_loss 

    def train_decoder(self, obs):
        """
        Train the decoder to reconstruct the input observations.
        """
        with tf.GradientTape() as tape:
            h = tf.stop_gradient(self.encode(obs, ema=True))
            reconst_obs = self.decoder(h)
            reconst_loss = tf.reduce_mean(tf.keras.metrics.mean_squared_error(obs, reconst_obs))
        grads = tape.gradient(reconst_loss, self.decoder.model.trainable_variables)
        self.decoder.optimizer.apply_gradients(zip(grads, self.decoder.model.trainable_variables))
        return reconst_loss

    def train_feature_predictor(self, obs, aug_obs):
        """
        Train the feature predictor to predict the features of the input observations.
        """

        h = tf.stop_gradient(self.encode(obs, ema=True))       # target for predictor
        h_aug = tf.stop_gradient(self.encode(aug_obs, ema=True)) # input to predictor
        
        consy_loss = self.feature_predictor.train(h_aug, h)
        return consy_loss

    def train_encoder(self, x_a, x_pos, obs, alpha_1=0.33, alpha_2=0.33, alpha_3=0.33):
        """ train the model on the data
        Arguments:
        x_a: (B, H, W, C) - anchor observations after augmentation
        x_pos: (B, H, W, C) - positive observations after augmentation
        obs: (B, H, W, C) - original observations before augmentation
        """
        # update W to minimize cosine similarity between z_a and z_pos
        with tf.GradientTape() as tape1:
            z_a = self.encode(x_a)
            z_pos = self.encode(x_pos, ema=True)
            logits, labels = self.compute_logits(z_a, z_pos)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, axis=-1)
            train_wts = [self.W]
        gradients_1 = tape1.gradient(loss, train_wts)
        self.optimizer.apply_gradients(zip(gradients_1, train_wts))

        # update encoder by incorporating all the three losses:
        # 1. reconstruction loss
        # 2. consistency loss
        # 3. contrastive loss
        with tf.GradientTape() as enc_tape:
            z_a = self.encode(x_a)
            z_pos = self.encode(x_pos, ema=True)
            logits, labels = self.compute_logits(z_a, z_pos)

            # Contrastive loos
            cont_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # reconstruction loss
            if self.include_reconst_loss:
                h = self.encoder(obs)
                rec_obs = self.decoder(h)
                reconst_loss = tf.reduce_mean(tf.keras.metrics.mean_squared_error(obs, rec_obs))
                loss = cont_loss + reconst_loss 
            else:
                reconst_loss = 0

            # consistency loss
            if self.include_consistency_loss:
                h = self.encode(obs)   # feature from original observation
                pred_feat = self.feature_predictor(h)
                pred_feat_norm = tf.math.l2_normalize(pred_feat, axis=1)

                target_feat = self.encode(x_a, ema=True) # feature from augmented observation
                target_feat_norm = tf.math.l2_normalize(target_feat, axis=-1)

                consistency_loss = tf.reduce_mean(tf.keras.metrics.mean_squared_error(target_feat_norm, pred_feat_norm))
            else:
                consistency_loss = 0


            total_loss = alpha_1 *  cont_loss + \
                            alpha_2 * reconst_loss + \
                                alpha_3  * consistency_loss

        enc_wts = self.encoder.model.trainable_variables
        enc_grads = enc_tape.gradient(total_loss, enc_wts)
        self.encoder.optimizer.apply_gradients(zip(enc_grads, enc_wts))

        return total_loss.numpy()

    def save_weights(self, filename):
        with open(str(filename), 'wb') as f:
            pickle.dump(self.W, f)

    def load_weights(self, filename):
        with open(str(filename), 'rb') as f:
            self.W = pickle.load(f)


    
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
                ac_train_freq=2,
                enc_train_freq=1,
                target_update_freq=2,           
                include_reconst_loss=False,
                include_consistency_loss=False,
                frozen_encoder=False,
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
            self.include_reconst_loss = include_reconst_loss
            self.include_consistency_loss = include_consistency_loss
            self.frozen_encoder = frozen_encoder

            # update frequencies
            self.ac_train_freq = ac_train_freq
            self.enc_train_freq = enc_train_freq
            self.eval_freq = eval_freq
            self.target_update_freq = target_update_freq
            self.init_steps = init_steps
            
            assert len(self.state_size) == 3, 'image observation of shape (h, w, c)'

            self.obs_shape = (self.cropped_img_size, self.cropped_img_size, self.state_size[2])

            # Create a common encoder network to be shared 
            # between the actor and the critic networks
            self.encoder = Encoder(self.obs_shape, self.feature_dim)



            # Actor
            self.actor = CurlActor(
                    state_size=self.obs_shape, 
                    action_size=self.action_shape,
                    action_upper_bound=self.action_upper_bound,
                    encoder_feature_dim=self.feature_dim,
                    encoder=self.encoder, # pass the encoder network
                    frozen_encoder=self.frozen_encoder
                    )

            # Critic
            self.critic = CurlCritic(
                    state_size=self.obs_shape,
                    action_size=self.action_shape,
                    encoder_feature_dim=self.feature_dim,
                    learning_rate=self.lr,
                    encoder = self.encoder, # pass the encoder network
                    frozen_encoder=self.frozen_encoder
            )

            # target critic
            # make sure, encoder for critic and target have same shape/size
            self.target_critic = CurlCritic(
                    state_size=self.obs_shape,
                    action_size=self.action_shape,
                    encoder_feature_dim=self.feature_dim,
                    learning_rate=self.lr,
                    #encoder=self.encoder,  # pass the encoder network 
                    model_name='target_critic'
            )

            # Initially critic & target critic share weights
            self.target_critic.model.set_weights(self.critic.model.get_weights())


            # CURL agent
            self.curl = CURL(
                obs_shape=self.obs_shape,
                z_dim = self.feature_dim,
                batch_size=self.batch_size,
                critic=self.critic,
                target_critic=self.target_critic,
                include_reconst_loss=self.include_reconst_loss,
                include_consistency_loss=self.include_consistency_loss
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

        with tf.GradientTape() as tape: 
            q1, q2 = self.critic(states, actions)

            pi_a, log_pi_a = self.actor.policy(next_states)

            q1_target, q2_target = self.target_critic(next_states, pi_a)

            min_q_target = tf.minimum(q1_target, q2_target)

            soft_q_target = min_q_target - self.alpha * tf.reduce_sum(log_pi_a, axis=-1, keepdims=True) 

            y = tf.stop_gradient(rewards + self.gamma * soft_q_target * (1 - dones)) 

            critic_loss = tf.reduce_mean(tf.square(y - q1)) + tf.reduce_mean(tf.square(y - q2))
            variables = self.critic.model.trainable_variables
        grads = tape.gradient(critic_loss, variables)
        self.critic.optimizer.apply_gradients(zip(grads, variables))
        return critic_loss.numpy()

    def update_actor_network(self, states):
        with tf.GradientTape() as tape:
            pi_a, log_pi_a = self.actor.policy(states)
            q1, q2 = self.critic(states, pi_a)
            min_q = tf.minimum(q1, q2)
            soft_q = min_q - self.alpha * log_pi_a
            actor_loss = tf.reduce_mean(soft_q)
        grads = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.model.trainable_variables))
        return actor_loss.numpy()

    def update_target_networks(self):
        for wt_target, wt_source in zip(self.target_critic.model.trainable_variables, 
                                self.critic.model.trainable_variables):
            wt_target = self.polyak * wt_target + (1 - self.polyak) * wt_source

        # Probably, this is redundant as encoder is getting updated in the critic network
        for wt_target, wt_source in zip(self.curl.encoder_target.model.trainable_variables,
                                    self.curl.encoder.model.trainable_variables):
            wt_target = self.polyak * wt_target + (1 - self.polyak) * wt_source

    def create_image_pairs(self, states, next_states, aug_method='crop'):

        if aug_method == 'crop':
            obs_a = random_crop(states, self.cropped_img_size)
            obs_p = random_crop(states, self.cropped_img_size)
            obs_n = random_crop(next_states, self.cropped_img_size)
        else:
            raise Exception('Unknown Observation Method')
        return obs_a, obs_p, obs_n

    def train_actor_critic(self):
        # sample a batch of data
        states, actions, rewards, next_states, dones = self.buffer.sample()

        states = random_crop(states, self.cropped_img_size)
        next_states = random_crop(next_states, self.cropped_img_size)


        # convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # update actor
        actor_loss = self.update_actor_network(states)

        # update the critic
        critic_loss = self.update_critic_networks(states,
                            actions, rewards, next_states, dones) 
                            
        # update alpha
        alpha_loss = self.update_alpha(states)

        return actor_loss, critic_loss, alpha_loss

    def train_encoder(self): # this needs to change ....
        # sample a batch of data
        states, _, _, next_states, _ = self.buffer.sample()

        # apply data augmentation to create image pairs
        obs_a, obs_p, _ = self.create_image_pairs(states, next_states)

        # observations before augmentation (cropped to fit the size)
        org_obs = center_crop_image(states, self.cropped_img_size)

        # update the encoder
        curl_loss = self.curl.train(obs_a, obs_p, org_obs)

        return  curl_loss

    def validate(self, env, max_eps=20):
        '''
        evaluate model performance
        '''
        ep_reward_list = []
        for _ in range(max_eps):
            state = env.reset()  # normalized floating point pixel obsvn
            t = 0
            ep_reward = 0
            while True:
                cropped_state = center_crop_image(state, out_h=self.cropped_img_size)
                action, _ = self.sample_action(cropped_state)
                next_state, reward, done, _ = env.step(action)

                # convert negative reward to positive reward 
                reward = 1 if reward == 0 else 0
                state = next_state 
                ep_reward += reward 
                t += 1
                if done:
                    ep_reward_list.append(ep_reward)
                    break
        # outside the loop
        mean_ep_reward = np.mean(ep_reward_list)
        return mean_ep_reward

    def run(self, env, max_train_steps=100000, WB_LOG=False):
        """ Main Training Loop"""

        actor_loss, critic_loss, alpha_loss, curl_loss = [], [], [], []
        episode, ep_reward, val_score, done = 0, 0, 0, False 
        ep_rewards, val_scores, alpha_losses, curl_losses = [], [], [], []
        actor_losses, critic_losses = [], []

        # initial state
        state = env.reset() # normalized floating point pixel obsvn
        for step in range(max_train_steps):

            # process the state 
            cropped_state = center_crop_image(state, out_h=self.cropped_img_size)

            # sample action
            if step < self.init_steps:
                action = env.action_space.sample()
            else:
                action, _ = self.sample_action(cropped_state)

            # take action
            next_state, reward, done, _ = env.step(action)

            # convert negative reward to positive reward 
            reward = 1 if reward == 0 else 0
            ep_reward += reward

            # store transition in buffer
            self.buffer.record([state, action, reward, next_state, done])

            # train actor & critic networks
            if step > self.init_steps and step % self.ac_train_freq == 0:
                actor_loss, critic_loss, alpha_loss = self.train_actor_critic() 
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                alpha_losses.append(alpha_loss)

            # train encoder using contrastive learning
            if step > self.init_steps and step % self.enc_train_freq == 0:
                curl_loss = self.train_encoder()
                curl_losses.append(curl_loss)

            # update target networks
            if step > self.init_steps and step % self.target_update_freq == 0:
                self.update_target_networks()

            # evaluate model performance
            if step > self.init_steps and step % self.eval_freq == 0:
                val_score = self.validate(env)
                val_scores.append(val_score)


            # prepare for next iteration
            state = next_state

            # logging
            if WB_LOG:
                wandb.log({
                    'step': step,
                    'episodes': episode,
                    'ep_reward': ep_reward,
                    'mean_ep_reward': np.mean(ep_rewards),
                    'mean_val_score': np.mean(val_scores),
                    'mean_critic_loss': np.mean(critic_losses),
                    'mean_alpha_loss': np.mean(alpha_losses),
                    'mean_actor_loss': np.mean(actor_losses),
                    'mean_curl_loss': np.mean(curl_losses),
                })

            if done: # end of episode
                ep_rewards.append(ep_reward)
                ep_reward = 0
                state = env.reset()
                episode += 1
                done = False 

        # end of training
        env.close()
        print('Mean Episode Reward: {} over {} episodes'.format(np.mean(ep_rewards), episode))

    def save_model(self, save_path):
        '''
        save model weights
        '''
        self.actor.save_weights(save_path + 'actor.h5')
        self.critic_1.save_weights(save_path + 'critic_1.h5')
        self.critic_target_1.save_weights(save_path + 'critic_target_1.h5')
        self.critic_2.save_weights(save_path + 'critic_2.h5')
        self.critic_target_2.save_weights(save_path + 'critic_target_2.h5')
        self.encoder.save_weights(save_path + 'encoder.h5')
        self.curl.save_weights(save_path + 'curl.h5')

    def load_weights(self, save_path):
        '''
        load model weights
        '''
        self.actor.load_weights(save_path + 'actor.h5')
        self.critic_1.load_weights(save_path + 'critic_1.h5')
        self.critic_target_1.load_weights(save_path + 'critic_target_1.h5')
        self.critic_2.load_weights(save_path + 'critic_2.h5')
        self.critic_target_2.load_weights(save_path + 'critic_target_2.h5')
        self.encoder.load_weights(save_path + 'encoder.h5')
        self.curl.load_weights(save_path + 'curl.h5')








            




    
