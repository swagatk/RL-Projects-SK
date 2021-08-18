'''
Soft Actor Critic Algorithm
Original Source: https://github.com/shakti365/soft-actor-critic

- policy function is now a part of the actor network. 
- tf.GradientTape() part of the actor/critic networks are modified to ensure proper flow of gradients. This is something that should be
    replicated to other algorithms such as PPO, IPG, DDPG, TD3 etc. (To do)
- Two versions of SACAgent class definitions are provided. The first version takes 'env' as an input argument which makes it difficult to account
    for changes in environment specifications. The second version SACAgent2 keeps the environment separate from agent definition making it easier
    for the user to pre-process environment input/output signals before passing them to the agent. This is something we will replicate
    in future algorithms. 
- SACAgent2 will supersede SACAgent in the future. 

'''
import numpy as np
import tensorflow as tf
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
from common.FeatureNet import FeatureNetwork, AttentionFeatureNetwork
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
        self.model = self._build_net(trainable=True)    # returns mean action
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # additional output
        logstd = tf.Variable(np.zeros(shape=self.action_size, dtype=np.float32))
        self.model.logstd = logstd
        self.model.trainable_variables.append(logstd)

    def _build_net(self, trainable=True):
        # input is a stack of 1-channel YUV images
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)
        state_input = tf.keras.layers.Input(shape=self.state_size)

        if self.feature_model is None:
            f = tf.keras.layers.Dense(128, activation='relu', trainable=trainable)(state_input)
        else:
            f = self.feature_model(state_input)

        f = tf.keras.layers.Dense(128, activation='relu', trainable=trainable)(f)
        f = tf.keras.layers.Dense(64, activation="relu", trainable=trainable)(f)
        net_out = tf.keras.layers.Dense(self.action_size[0], activation='tanh',
                                        kernel_initializer=last_init, trainable=trainable)(f)
        net_out = net_out * self.upper_bound  # element-wise product
        model = tf.keras.Model(state_input, net_out, name='actor')
        model.summary()
        tf.keras.utils.plot_model(model, to_file='actor_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        # input is a tensor
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std

    def policy(self, state):
        # input: tensor
        # output: tensor
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))

        pi = tfp.distributions.Normal(mean, std)
        action_ = pi.sample()
        log_pi_ = pi.log_prob(action_)
        action = tf.clip_by_value(action_, -self.upper_bound, self.upper_bound)

        if tf.rank(action) < 1:     # scalar
            log_pi_a = log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)
        elif 1 <= tf.rank(action) < 2:  # vector
            log_pi_a = tf.reduce_sum((log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)), axis=0, keepdims=True)
        else:   # matrix
            log_pi_a = tf.reduce_sum((log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)), axis=1, keepdims=True)

        return action, log_pi_a

    def train(self, states, alpha, critic1, critic2):
        with tf.GradientTape() as tape:
            actions, log_pi_a = self.policy(states)
            q1 = critic1(states, actions)
            q2 = critic2(states, actions)
            min_q = tf.minimum(q1, q2)
            soft_q = min_q - alpha * log_pi_a
            actor_loss = -tf.reduce_mean(soft_q)
        # outside gradient tape block
        actor_wts = self.model.trainable_variables
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
        q_value = tf.squeeze(self.model([state, action]))
        return q_value

    def train(self, states, actions, rewards, next_states, dones, actor,
              target_critic1, target_critic2, alpha):
        with tf.GradientTape() as tape:
            # Get Q estimates using actions from replay buffer
            q_values = tf.squeeze(self.model([states, actions]))

            # Sample actions from the policy network for next states
            a_next, log_pi_a = actor.policy(next_states)

            # Get Q value estimates from target Q networks
            q1_target = target_critic1(next_states, a_next)
            q2_target = target_critic2(next_states, a_next)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - alpha * log_pi_a

            y = tf.stop_gradient(rewards * self.gamma * (1 - dones) * soft_q_target)
            critic_loss = tf.math.reduce_mean(tf.square(y - q_values))
            critic_wts = self.model.trainable_variables
        # outside gradient tape
        critic_grad = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grad, critic_wts))
        return critic_loss.numpy()

    def train2(self, state_batch, action_batch, y):
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
    def __init__(self, env, SEASONS, success_value, epochs,
                 training_batch, batch_size, buffer_capacity, lr_a=0.0003, lr_c=0.0003, 
                 gamma=0.99, tau=0.995, alpha=0.2, use_attention=None, 
                 filename=None, wb_log=False, chkpt_freq=None, path='./'):
        self.env = env
        self.action_size = self.env.action_space.shape
        self.state_size = self.env.observation_space.shape
        self.upper_bound = np.squeeze(self.env.action_space.high)

        self.seasons = SEASONS
        self.success_value = success_value
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.epochs = epochs
        self.training_batch = training_batch    # training steps in each season
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_entropy = -tf.constant(np.prod(self.action_size), dtype=tf.float32)
        self.gamma = gamma                  # discount factor
        self.tau = tau                      # polyak averaging factor
        self.use_attention = use_attention
        self.filename = filename
        self.WB_LOG = wb_log
        self.path = path
        self.episodes = 0                   # total number of episodes
        self.chkpt_freq = chkpt_freq                  # save chkpts

        if len(self.state_size) == 3:
            self.image_input = True     # image input
        elif len(self.state_size) == 1:
            self.image_input = False        # vector input
        else:
            raise ValueError("Input can be a vector or an image")

        # extract features from input images
        if self.image_input:        # input is an image
            print('Currently Attention handles only image input')
            self.feature = FeatureNetwork(self.state_size, self.use_attention, self.lr_a)
        else:       # non-image input
            print('You have selected a non-image input.')
            self.feature = None

        # Actor network
        self.actor = SACActor(self.state_size, self.action_size, self.upper_bound,
                              self.lr_a, self.feature)

        # create two critics
        self.critic1 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.critic2 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)

        # create two target critics
        self.target_critic1 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.target_critic2 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)

        # create alpha as a trainable variable
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr_a)

        # Buffer for off-policy training
        self.buffer = Buffer(self.buffer_capacity, self.batch_size)

    def policy(self, state):
        # input: numpy array
        # output: numpy array
        if state.ndim < len(self.state_size) + 1:      # single sample
            tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        else:
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)

        action, log_pi = self.actor.policy(tf_state)   # returns tensors
        return action.numpy(), log_pi.numpy()

    def update_alpha(self, states):
        # input: tensor
        with tf.GradientTape() as tape:
            # sample actions from the policy for the current states
            _, log_pi_a = self.actor.policy(states)
            alpha_loss = - tf.reduce_mean(self.alpha * (log_pi_a + self.target_entropy))
        # outside gradient tape block
        variables = [self.alpha]
        grads = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))
        return alpha_loss.numpy()

    def update_target_networks(self):
        target_wts1 = np.array(self.target_critic1.model.get_weights())
        wts1 = np.array(self.critic1.model.get_weights())
        target_wts2 = np.array(self.target_critic2.model.get_weights())
        wts2 = np.array(self.critic2.model.get_weights())

        target_wts1 = self.tau * target_wts1 + (1 - self.tau) * wts1
        target_wts2 = self.tau * target_wts2 + (1 - self.tau) * wts2

        self.target_critic1.model.set_weights(target_wts1)
        self.target_critic2.model.set_weights(target_wts2)

    def update_q_networks(self, states, actions, rewards, next_states, dones):

        pi_a, log_pi_a = self.actor.policy(next_states) # output: tensor

        q1_target = self.target_critic1(next_states, pi_a) # input: tensor
        q2_target = self.critic2(next_states, pi_a)

        min_q_target = tf.minimum(q1_target, q2_target)

        soft_q_target = min_q_target  - self.alpha * log_pi_a  

        y = rewards + self.gamma * (1 - dones) * soft_q_target

        c1_loss = self.critic1.train2(states, actions, y)
        c2_loss = self.critic2.train2(states, actions, y)

        mean_c_loss = np.mean([c1_loss, c2_loss])
        return mean_c_loss 

    def train(self, CRIT_T2=True):
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
            if not CRIT_T2: 
                c1_loss = self.critic1.train(states, actions, rewards, next_states, dones,
                                            self.actor, self.target_critic1, self.target_critic2,
                                            self.alpha)

                c2_loss = self.critic2.train(states, actions, rewards, next_states, dones,
                                            self.actor, self.target_critic1, self.target_critic2,
                                            self.alpha)

                critic_loss = np.mean([c1_loss, c2_loss])
            else:
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
            state = np.asarray(state, dtype=np.float32) / 255.0

            t = 0
            ep_reward = 0
            while True:
                action, _ = self.policy(state)
                next_obsv, reward, done, _ = env.step(action)
                next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0

                state = next_state
                ep_reward += reward
                t += 1
                if done:
                    ep_reward_list.append(ep_reward)
                    break
        # outside for loop
        mean_ep_reward = np.mean(ep_reward_list)
        return mean_ep_reward

    def run(self, train_freq=20):
        if self.filename is not None:
            self.filename = uniquify(self.path + self.filename)


        # initial state
        state = self.env.reset()
        state = np.asarray(state, dtype=np.float32) / 255.0

        start = datetime.datetime.now()
        val_scores = []         # validation scores
        best_score = -np.inf
        ep_lens = []            # episodic length
        ep_scores = []          # episodic scores
        s_scores = []           # season scores
        ep_actor_losses = []    # actor losses 
        ep_critic_losses = []   # critic losses 
        self.episodes = 0       # global episode count
        for s in range(self.seasons):
            s_score = 0         # season score
            ep_cnt = 0          # episodes in each season
            ep_len = 0          # len of each episode
            ep_score = 0        # score of each episode
            done = False    
            for t in range(self.training_batch):
                action, _ = self.policy(state)
                next_obs, reward, done, _ = self.env.step(action)
                next_state = np.asarray(next_obs, dtype=np.float32) / 255.0

                # store in replay buffer for off-policy training
                self.buffer.record([state, action, reward, next_state, done])

                state = next_state
                ep_score += reward
                ep_len += 1

                if done:
                    s_score += ep_score
                    ep_cnt += 1
                    self.episodes += 1      # total episode count
                    ep_scores.append(ep_score)
                    ep_lens.append(ep_len)
                    
                    # train after each episode
                    if self.episodes % train_freq == 0:
                        a_loss, c_loss, alpha_loss = self.train()
                        ep_actor_losses.append(a_loss)
                        ep_critic_losses.append(c_loss)

                        if self.WB_LOG:
                            wandb.log({'ep_actor_loss' : a_loss,
                            'ep_critic_loss' : c_loss,
                            'ep_alpha_loss' : alpha_loss})

                    if self.WB_LOG:
                        wandb.log({
                            'Episodes' : self.episodes, 
                            'mean_ep_score': np.mean(ep_scores),
                            'mean_ep_len' : np.mean(ep_lens)})
                    
                    # prepare for next episode
                    state = np.asarray(self.env.reset(), dtype=np.float32) / 255.0
                    ep_len, ep_score = 0, 0
                    done = False
                # done block ends here
            # end of one season

            s_score = np.mean(ep_scores[-ep_cnt : ])
            s_scores.append(s_score)
            mean_ep_score = np.mean(ep_scores)
            mean_ep_len = np.mean(ep_lens)
            mean_s_score = np.mean(s_scores)
            mean_actor_loss = np.mean(ep_actor_losses[-ep_cnt:])
            mean_critic_loss = np.mean(ep_critic_losses[-ep_cnt:])

            if mean_s_score > best_score:
                best_model_path = self.path + 'best_model/'
                os.makedirs(best_model_path, exist_ok=True)
                self.save_model(best_model_path)
                print('Season: {}, Update best score: {}-->{}, Model saved!'.format(s, best_score, mean_s_score))
                best_score = mean_s_score
            
            # run validation once after each season
            val_score = self.validate(self.env)
            val_scores.append(val_score)
            mean_val_score = np.mean(val_scores)
            print('season: {}, Validation Score: {}, Mean Validation Score: {}' \
                    .format(s, val_score, mean_val_score))

            if self.WB_LOG:
                wandb.log({'Season Score' : s_score, 
                            'Mean Season Score' : mean_s_score,
                            'Actor Loss' : mean_actor_loss,
                            'Critic Loss' : mean_critic_loss,
                            'Mean episode length' : mean_ep_len,
                            'val_score' : val_score,
                            'mean_val_score' : mean_val_score,
                            'ep_cnt' : ep_cnt,
                            'Season' : s})

            if self.chkpt_freq is not None and s % self.chkpt_freq == 0:
                chkpt_path = self.path + 'chkpt_{}/'.format(s)
                os.makedirs(chkpt_path, exist_ok=True)
                self.save_model(chkpt_path)

            if self.filename is not None:
                with open(self.filename, 'a') as file:
                    file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                            .format(s, self.episodes, mean_ep_len,
                                    s_score, mean_s_score, ep_cnt, mean_actor_loss, mean_critic_loss, alpha_loss,
                                    val_score, mean_val_score))

            if self.success_value is not None:
                if best_score > self.success_value:
                    print('Problem is solved in {} episodes with score {}'.format(s, best_score))
                    print('Mean Episodic score: {}'.format(mean_ep_score))
                    break
        # end of season-loop
        end = datetime.datetime.now()
        print('Time to Completion: {}'.format(end - start))
        self.env.close()
        print('Mean episodic score over {} episodes: {:.2f}'.format(self.episodes, np.mean(ep_scores)))

    def save_model(self, save_path):
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

class SACAgent2:
    # the environment variable is not a part of this class
    def __init__(self, state_size, action_size, action_upper_bound, epochs,
                  batch_size, buffer_capacity, lr_a=0.0003, lr_c=0.0003,
                 gamma=0.99, tau=0.995, alpha=0.2, use_attention=False):
        self.action_size = action_size
        self.state_size = state_size
        self.upper_bound = action_upper_bound
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_entropy = -tf.constant(np.prod(self.action_size), dtype=tf.float32)
        self.gamma = gamma                  # discount factor
        self.tau = tau                      # polyak averaging factor
        self.use_attention = use_attention

        if len(self.state_size) == 3:
            self.image_input = True     # image input
        elif len(self.state_size) == 1:
            self.image_input = False        # vector input
        else:
            raise ValueError("Input can be a vector or an image")

        # Select a suitable feature extractor
        if self.use_attention and self.image_input:   # attention + image input
            print('Currently Attention handles only image input')
            self.feature = AttentionFeatureNetwork(self.state_size, self.lr_a)
        elif self.use_attention is False and self.image_input is True:  # image input
            print('You have selected an image input')
            self.feature = FeatureNetwork(self.state_size, self.lr_a)
        else:       # non-image input
            print('You have selected a non-image input.')
            self.feature = None

        # create actor for learning policy
        self.actor = SACActor(self.state_size, self.action_size, self.upper_bound,
                              self.lr_a, self.feature)

        # create two critics
        self.critic1 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.critic2 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)

        # create two target critics
        self.target_critic1 = SACCritic(self.state_size, self.action_size,
                                        self.lr_c, self.gamma, self.feature)
        self.target_critic2 = SACCritic(self.state_size, self.action_size,
                                        self.lr_c, self.gamma, self.feature)

        # create alpha as a trainable variable
        # This is the entropy coefficient required for soft target
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(self.lr_a)

        # Buffer for off-policy training
        self.buffer = Buffer(self.buffer_capacity, self.batch_size)

    def policy(self, state):
        if state.ndim < len(self.state_size) + 1:      # single instance
            tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        else:       # for a batch of samples
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)

        action, log_pi = self.actor.policy(tf_state) # returns tensors
        return action.numpy(), log_pi.numpy()

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
        target_wts1 = np.array(self.target_critic1.model.get_weights())
        wts1 = np.array(self.critic1.model.get_weights())
        target_wts2 = np.array(self.target_critic2.model.get_weights())
        wts2 = np.array(self.critic2.model.get_weights())

        target_wts1 = self.tau * target_wts1 + (1 - self.tau) * wts1
        target_wts2 = self.tau * target_wts2 + (1 - self.tau) * wts2

        self.target_critic1.model.set_weights(target_wts1)
        self.target_critic2.model.set_weights(target_wts2)

    def update_q_networks(self, states, actions, rewards, next_states, dones):

        pi_a, log_pi_a = self.actor.policy(next_states) # output: tensor

        q1_target = self.target_critic1(next_states, pi_a) # input: tensor
        q2_target = self.critic2(next_states, pi_a)

        min_q_target = tf.minimum(q1_target, q2_target)

        soft_q_target = min_q_target  - self.alpha * log_pi_a  

        y = rewards + self.gamma * (1 - dones) * soft_q_target

        c1_loss = self.critic1.train2(states, actions, y)
        c2_loss = self.critic2.train2(states, actions, y)

        mean_c_loss = np.mean([c1_loss, c2_loss])
        return mean_c_loss 

    def replay(self, CRIT_T2=True):
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

            # update Q (Critic) network weights
            if CRIT_T2: 
                c1_loss = self.critic1.train(states, actions, rewards, next_states, dones,
                                             self.actor, self.target_critic1, self.target_critic2,
                                             self.alpha)

                c2_loss = self.critic2.train(states, actions, rewards, next_states, dones,
                                             self.actor, self.target_critic1, self.target_critic2,
                                             self.alpha)
                critic_loss = np.mean([c1_loss, c2_loss])
            else:
                critic_loss = self.update_q_networks(states, actions, rewards, next_states, dones)

            # update (actor) policy networks
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

        return  mean_actor_loss, mean_critic_loss, mean_alpha_loss

    def save_model(self, save_path):
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

