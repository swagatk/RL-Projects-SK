'''
Soft Actor Critic Algorithm
Original Source: https://github.com/shakti365/soft-actor-critic

- policy function is now a part of the actor network. 
- tf.GradientTape() part of the actor/critic networks are modified to ensure proper flow of gradients. This is something that should be
    replicated to other algorithms such as PPO, IPG, DDPG, TD3 etc. (To do)

'''
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import extract_image_patches_eager_fallback
import tensorflow_probability as tfp
from tensorflow.keras import layers
import os
import datetime
import random
from collections import deque
import wandb
import sys

# Add the current folder to python's import path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

# Local imports
from common.FeatureNet import FeatureNetwork
from common.buffer import Buffer
from common.utils import uniquify


###############
# ACTOR NETWORK
###############

class SACActor:
    def __init__(self, state_size, action_size, upper_bound,
                 learning_rate, feature):
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.upper_bound = upper_bound

        # create NN models
        self.feature_model = feature
        self.model = self._build_net()    # returns mean action
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        # input is a stack of 1-channel YUV images
        s_inp = tf.keras.layers.Input(shape=self.state_size)

        if self.feature_model is None:
            f = tf.keras.layers.Dense(128, activation='relu')(s_inp)
        else:
            f = self.feature_model(s_inp)

        f = tf.keras.layers.Dense(128, activation='relu')(f)
        f = tf.keras.layers.Dense(64, activation="relu")(f)
        mu = tf.keras.layers.Dense(self.action_size[0], activation='tanh')(f)
        log_sig = tf.keras.layers.Dense(self.action_size[0])(f)
        mu = mu * self.upper_bound  # scale to original range
        model = tf.keras.Model(s_inp, [mu, log_sig], name='actor')
        model.summary()
        tf.keras.utils.plot_model(model, to_file='actor_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        # input is a tensor
        mean, log_sigma = self.model(state)
        std = tf.math.exp(log_sigma)
        return mean, std

    def policy(self, state):
        # input: tensor
        # output: tensor
        mean, std = self.__call__(state)

        pi = tfp.distributions.Normal(mean, std)
        action_ = pi.sample()
        log_pi_ = pi.log_prob(action_)

        action = tf.clip_by_value(action_, -self.upper_bound, self.upper_bound)

        log_pi_a = log_pi_ - tf.reduce_sum(tf.math.log( 1 - action ** 2 + 1e-16), axis=-1, keepdims=True)

        # if tf.rank(action) < 1:     # scalar
        #     log_pi_a = log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)
        # elif 1 <= tf.rank(action) < 2:  # vector
        #     log_pi_a = tf.reduce_sum((log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)), axis=0, keepdims=True)
        # else:   # matrix
        #     log_pi_a = tf.reduce_sum((log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)), axis=1, keepdims=True)

        return action, log_pi_a

    def train(self, states, alpha, critic1, critic2):
        with tf.GradientTape() as tape:
            actions, log_pi_a = self.policy(states)
            q1 = critic1(states, actions)
            q2 = critic2(states, actions)
            min_q = tf.minimum(q1, q2)
            soft_q = min_q - alpha * log_pi_a
            actor_loss = -tf.reduce_mean(soft_q) # maximize this loss 
            actor_wts = self.model.trainable_variables
        # outside gradient tape block
        actor_grads = tape.gradient(actor_loss, actor_wts)
        self.optimizer.apply_gradients(zip(actor_grads, actor_wts))
        return actor_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class SACCritic:
    """
    Approximates Q(s,a) function
    """
    def __init__(self, state_size, action_size, learning_rate,
                 gamma, feature_model):
        self.state_size = state_size    # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.gamma = gamma          # discount factor

        # create NN model
        self.feature = feature_model
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        state_input = layers.Input(shape=self.state_size)

        if self.feature is None:
            f = layers.Dense(128, activation='relu')(state_input)
        else:
            f = self.feature(state_input)

        state_out = layers.Dense(32, activation='relu')(f)
        state_out = layers.Dense(32, activation='relu')(state_out)

        # action as input
        action_input = layers.Input(shape=self.action_size)
        action_out = layers.Dense(32, activation='relu')(action_input)

        concat = layers.Concatenate()([state_out, action_out])
        out = layers.Dense(128, activation='relu')(concat)
        out = layers.Dense(64, activation='relu')(out)
        net_out = layers.Dense(1)(out)
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=net_out)
        model.summary()
        tf.keras.utils.plot_model(model, to_file='critic_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state, action):
        # input: tensors
        q_value = self.model([state, action])
        return q_value

    def train(self, state_batch, action_batch, y):
        with tf.GradientTape() as tape:
            q_values = self.model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.square(y - q_values))
            critic_wts = self.model.trainable_variables
        # outside gradient tape
        critic_grad = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grad, critic_wts))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class SACAgent:
    def __init__(self, state_size, action_size, action_upper_bound,
                buffer_capacity=100000, batch_size=128, epochs=50, 
                learning_rate=0.0003, alpha=0.2, gamma=0.99,
                polyak=0.995, use_attention=None, 
                filename=None, wb_log=False, path='./'):
                
        self.action_size = action_size 
        self.state_size = state_size 
        self.upper_bound = action_upper_bound 

        self.lr = learning_rate
        self.epochs = epochs 
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_entropy = -tf.constant(np.prod(self.action_size), dtype=tf.float32)
        self.gamma = gamma                  # discount factor
        self.polyak = polyak                      # polyak averaging factor
        self.use_attention = use_attention
        self.filename = filename
        self.WB_LOG = wb_log
        self.path = path
        self.global_steps = 0       # global step count

        if len(self.state_size) == 3:   # image input
            self.image_input = True     
        elif len(self.state_size) == 1: # vector input
            self.image_input = False        
        else:
            raise ValueError("Input can be a vector or an image")

        # extract features from input images
        if self.image_input:        # input is an image
            print('Currently Attention handles only image input')
            self.feature = FeatureNetwork(self.state_size, self.use_attention, self.lr)
        else:       # non-image input
            print('You have selected a non-image input.')
            self.feature = None

        # Actor network
        self.actor = SACActor(self.state_size, self.action_size, self.upper_bound,
                              self.lr, self.feature)

        # create two critics
        self.critic1 = SACCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.feature)
        self.critic2 = SACCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.feature)

        # create two target critics
        self.target_critic1 = SACCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.feature)
        self.target_critic2 = SACCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.feature)

        # create alpha as a trainable variable
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(self.lr)

        # Buffer for off-policy training
        self.buffer = Buffer(self.buffer_capacity, self.batch_size)

        # initially target networks share same weights as the original networks
        self.target_critic1.model.set_weights(self.critic1.model.get_weights())
        self.target_critic2.model.set_weights(self.critic2.model.get_weights())

    def sample_action(self, state):
        # input: numpy array
        # output: numpy array
        if state.ndim < len(self.state_size) + 1:      # single sample
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
        # outside gradient tape block
        variables = [self.alpha]
        grads = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))
        return alpha_loss.numpy()

    def update_target_networks(self):
        for theta_target, theta in zip(self.target_critic1.model.trainable_variables,
                    self.critic1.model.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta
        
        for theta_target, theta in zip(self.target_critic2.model.trainable_variables,
                    self.critic2.model.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta

    def update_q_networks(self, states, actions, rewards, next_states, dones):

        pi_a, log_pi_a = self.actor.policy(next_states) # output: tensor

        q1_target = self.target_critic1(next_states, pi_a) # input: tensor
        q2_target = self.critic2(next_states, pi_a)

        min_q_target = tf.minimum(q1_target, q2_target)

        soft_q_target = min_q_target  - self.alpha * tf.reduce_sum(log_pi_a, axis=1)  

        y = tf.stop_gradient(rewards + self.gamma * (1 - dones) * soft_q_target)

        c1_loss = self.critic1.train(states, actions, y)
        c2_loss = self.critic2.train(states, actions, y)

        mean_c_loss = np.mean([c1_loss, c2_loss])
        return mean_c_loss 

    def train(self):
        critic_losses, actor_losses, alpha_losses = [], [], []
        for epoch in range(self.epochs):
            # sample a minibatch from the replay buffer
            states, actions, rewards, next_states, dones = self.buffer.sample()

            # convert to tensors
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # update Q network weights
            critic_loss = self.update_q_networks(states, actions, rewards, next_states, dones)
                                    
            # update policy networks
            actor_loss = self.actor.train(states, self.alpha, self.critic1, self.critic2)

            # update entropy coefficient
            alpha_loss = self.update_alpha(states)

            # update target network weights
            self.update_target_networks()

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)
        # epoch loop ends here
        mean_critic_loss = np.mean(critic_losses)
        mean_actor_loss = np.mean(actor_losses)
        mean_alpha_loss = np.mean(alpha_losses)

        return mean_actor_loss, mean_critic_loss, mean_alpha_loss

    def validate(self, env, max_eps=50):
        ep_reward_list = []
        for ep in range(max_eps):
            state = env.reset()
            if self.image_input:
                state = np.asarray(state, dtype=np.float32) / 255.0


            t = 0
            ep_reward = 0
            while True:
                action, _ = self.sample_action(state)
                next_state, reward, done, _ = env.step(action)

                if self.image_input:
                    next_state = np.asarray(next_state, dtype=np.float32) / 255.0

                state = next_state
                ep_reward += reward
                t += 1
                if done:
                    ep_reward_list.append(ep_reward)
                    break
        # outside for loop
        mean_ep_reward = np.mean(ep_reward_list)
        return mean_ep_reward

    def run(self, env, max_episodes=1000):

        if self.filename is not None:
            self.filename = uniquify(self.path + self.filename)

        start = datetime.datetime.now()
        val_scores = []         # validation scores
        best_score = -np.inf
        ep_lens = []            # episodic length
        ep_scores = []          # episodic scores
        ep_actor_losses = []    # actor losses 
        ep_critic_losses = []   # critic losses 
        ep_alpha_losses = []

        for ep in range(max_episodes):

            state = env.reset()
            if self.image_input:
                state = np.asarray(state, dtype=np.float32) / 255.0

            ep_len = 0          # len of each episode
            ep_score = 0        # score of each episode
            done = False    
            while not done:
                action, _ = self.sample_action(state)
                next_state, reward, done, _ = self.env.step(action)

                if self.image_input:
                    next_state = np.asarray(next_state, dtype=np.float32) / 255.0

                # store in replay buffer for off-policy training
                self.buffer.record([state, action, reward, next_state, done])

                state = next_state
                ep_score += reward
                ep_len += 1
                self.global_steps += 1
            # while-loop ends here

            ep_scores.append(ep_score)
            ep_lens.append(ep_len)

            # train 
            a_loss, c_loss, alpha_loss = self.train()
            ep_actor_losses.append(a_loss)
            ep_critic_losses.append(c_loss)
            ep_alpha_losses.append(alpha_loss)

            # validate
            val_score = self.validate(self.env)
            val_scores.append(val_score)

            # log
            if self.WB_LOG:
                wandb.log({
                    'Episodes' : ep, 
                    'mean_ep_score': np.mean(ep_scores),
                    'mean_ep_len' : np.mean(ep_lens),
                    'ep_actor_loss': np.mean(ep_actor_losses),
                    'ep_critic_loss': np.mean(ep_critic_losses),
                    'ep_alpha_loss' : np.mean(ep_alpha_losses),
                    'mean_val_score': np.mean(val_scores)
                    })
                    
            if np.mean(ep_scores) > best_score:
                best_model_path = self.path + 'best_model/'
                self.save_model(best_model_path)
                best_score = np.mean(ep_scores) 
                print('Season: {}, Update best score: {:.2f}-->{:.2f}, Model saved!'.format(ep, best_score, np.mean(ep_scores)))

            if self.filename is not None:
                with open(self.filename, 'a') as file:
                    file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                            .format(ep, 
                            np.mean(ep_scores),
                            np.mean(ep_lens),
                            np.mean(ep_actor_losses),
                            np.mean(ep_critic_losses),
                            np.mean(ep_alpha_losses),
                            np.mean(val_scores)
                            ))

        # end of for loop
        end = datetime.datetime.now()
        print('Time to Completion: {}'.format(end - start))
        self.env.close()
        print('Mean episodic score over {} episodes: {:.2f}'.format(ep, np.mean(ep_scores)))
        print('Total number of steps:', self.global_steps)
        final_model_dir = self.path + 'final_model/'
        self.save_model(final_model_dir)

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        actor_file = save_path + 'sac_actor_wts.h5'
        critic1_file = save_path + 'sac_c1_wts.h5'
        critic2_file = save_path + 'sac_c2_wts.h5'
        target_c1_file = save_path + 'sac_c1t_wts.h5'
        target_c2_file = save_path + 'sac_c2t_wts.h5'
        self.actor.save_weights(actor_file)
        self.critic1.save_weights(critic1_file)
        self.critic2.save_weights(critic2_file)
        self.target_critic1.save_weights(target_c1_file)
        self.target_critic2.save_weights(target_c2_file)

    def load_model(self, load_path):
        actor_file = load_path + 'sac_actor_wts.h5'
        critic1_file = load_path + 'sac_c1_wts.h5'
        critic2_file = load_path + 'sac_c2_wts.h5'
        target_c1_file = load_path + 'sac_c1t_wts.h5'
        target_c2_file = load_path + 'sac_c2t_wts.h5'
        self.actor.load_weights(actor_file)
        self.critic1.load_weights(critic1_file)
        self.critic2.load_weights(critic2_file)
        self.target_critic1.load_weights(target_c1_file)
        self.target_critic2.load_weights(target_c2_file)

####################################

if __name__ == '__main__':

    import gym

    env = gym.make('MountainCarContinuous-v0')
    action_size = env.action_space.shape
    state_size = env.observation_space.shape
    action_upper_bound = env.action_space.high

    agent = SACAgent(env, state_size, action_size, action_upper_bound,
                            filename='mc_sac.txt')
    agent.run()