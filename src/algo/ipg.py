'''
Interpolated Policy Gradient (IPG) Method
- Originally implemented by Mr. Sampson Hayden
'''
import wandb
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import os
import datetime
import random
import sys 
from collections import deque

# Add the current folder to python's import path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

# Local imports
from common.FeatureNet import FeatureNetwork
from common.buffer import Buffer
from common.utils import uniquify

###############
# ACTOR NETWORK
###############

class IPGActor:
    def __init__(self, state_size, action_size, upper_bound,
                 lr, epsilon, feature):
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = lr
        self.epsilon = epsilon              # Clipping value
        self.upper_bound = upper_bound

        # create NN models
        self.feature_model = feature
        self.model = self._build_net(trainable=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # additions
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
        return mean, std # returns tensors

    def train(self, states, actions, advantages, old_pi, critic, b, s_batch):
        with tf.GradientTape() as tape:
            # on-policy ppo loss
            mean = tf.squeeze(self.model(states))
            std = tf.squeeze(tf.exp(self.model.logstd))
            pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(pi.log_prob(tf.squeeze(actions)) -
                          old_pi.log_prob(tf.squeeze(actions)))
            adv_stack = tf.stack([advantages for _ in range(self.action_size[0])], axis=1)  # shape(-1,3) check ..
            surr = ratio * adv_stack         # surrogate function
            clipped_surr = tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv_stack
            ppo_loss = tf.reduce_mean(tf.minimum(surr, clipped_surr))

            # off-policy ppo loss
            a_batch = tf.squeeze(self.model(s_batch))       # action estimate
            q_values = critic.model([s_batch, a_batch])     # q estimates
            sum_q_values = tf.reduce_sum(tf.reduce_mean(q_values))
            off_loss = (b / len(s_batch)) * sum_q_values
            actor_loss = -tf.reduce_sum(ppo_loss + off_loss)
            actor_wts = self.model.trainable_variables

        # outside gradient tape
        actor_grad = tape.gradient(actor_loss, actor_wts)
        self.optimizer.apply_gradients(zip(actor_grad, actor_wts))
        return actor_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class DDPGCritic:
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

    def train(self, state_batch, action_batch, y):
        with tf.GradientTape() as tape:
            critic_wts = self.model.trainable_variables
            critic_value = self.model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.square(y - critic_value))
        # outside gradient tape
        critic_grad = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grad, critic_wts))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class Baseline:
    def __init__(self, state_size, action_size, lr, feature):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.feature = feature
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        state_input = tf.keras.layers.Input(shape=self.state_size)

        if self.feature is None:
            f = tf.keras.layers.Dense(128, activation='relu')(state_input)
        else:
            f = self.feature(state_input)

        out = tf.keras.layers.Dense(128, activation='relu')(f)
        out = tf.keras.layers.Dense(64, activation='relu')(out)
        out = tf.keras.layers.Dense(32, activation='relu')(out)
        net_out = tf.keras.layers.Dense(1)(out)
        model = tf.keras.Model(inputs=state_input, outputs=net_out)
        model.summary()
        return model

    def train(self, states, returns):
        with tf.GradientTape() as tape:
            critic_wts = self.model.trainable_variables
            critic_values = self.model(states)
            critic_loss = tf.math.reduce_mean(tf.square(returns - critic_values))

        critic_grad = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grad, critic_wts))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class IPGAgent:
    def __init__(self, env, SEASONS, success_value, 
                 epochs, training_batch, batch_size, buffer_capacity, 
                 lr_a, lr_c, gamma, epsilon, lmbda,
                 stack_size=7,
                 use_attention=None, 
                 filename=None, wb_log=False, chkpt_freq=None, path='./'):
        self.env = env
        self.action_size = self.env.action_space.shape
        self.state_size = self.env.observation_space.shape
        self.upper_bound = self.env.action_space.high
        self.SEASONS = SEASONS
        self.episodes = 0           # global episode count 
        self.time_steps = 0         # global step count
        self.success_value = success_value
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.epochs = epochs
        self.training_batch = training_batch
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.use_attention = use_attention
        self.filename = filename
        self.WB_LOG = wb_log
        self.path = path
        self.chkpt_freq = chkpt_freq
        self.stack_size = stack_size

        if len(self.state_size) == 3:
            self.image_input = True     # image input
        elif len(self.state_size) == 1:
            self.image_input = False        # vector input
        else:
            raise ValueError("Input can be a vector or an image")

        # Buffer for off-policy training
        self.buffer = Buffer(self.buffer_capacity, self.batch_size)


        # extract features from input images
        if self.image_input:        # input is an image
            if self.stack_size > 1:
                self.state_size = (self.stack_size, ) + self.state_size

            self.feature = FeatureNetwork(self.state_size, self.use_attention, self.lr_a)
        else:       # non-image input
            print('You have selected a non-image input.')
            self.feature = None


        # create actor / critic models
        self.actor = IPGActor(self.state_size, self.action_size, self.upper_bound,
                           self.lr_a, self.epsilon, self.feature)
        self.critic = DDPGCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.baseline = Baseline(self.state_size, self.action_size, self.lr_c, self.feature)

    def policy(self, state, deterministic=False):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        mean, std = self.actor(tf_state)

        if deterministic:
            action = mean
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = pi.sample()
        #action = mean + np.random.uniform(-self.upper_bound, self.upper_bound, size=mean.shape) * std
        action = tf.clip_by_value(action, -self.upper_bound, self.upper_bound)
        return action.numpy()

    def extract_feature(self, state):
        f_state = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        feat = self.feature(f_state)
        return feat.numpy()

    def prepare_input(self, img_buffer):
        # input : list of images of shape: (h, w, c)
        # output: numpy array - a stack of images: (s, h, w, c)
        temp_list = []
        for i in range(self.stack_size):
            if i < len(img_buffer):
                temp_list.append(img_buffer[-1-i])      # fill from beginning
            else:
                temp_list.append(img_buffer[-1])        # last element

        stacked_img = np.stack(temp_list, axis=0)
        return stacked_img      # check the shape:  (stack_size, h, w, c)

    def compute_advantage(self, r_batch, s_batch, ns_batch, d_batch):
        # input: tensors
        gamma = self.gamma
        lmbda = self.lmbda
        s_values = tf.squeeze(self.baseline.model(s_batch)) # input: tensor
        ns_values = tf.squeeze(self.baseline.model(ns_batch))
        returns = []
        gae = 0     # generalized advantage estimate
        for i in reversed(range(len(r_batch))):
            delta = r_batch[i] + gamma * ns_values[i] * (1 - d_batch[i]) - s_values[i]
            gae = delta + gamma * lmbda * (1 - d_batch[i]) * gae
            returns. insert(0, gae + s_values[i])

        returns = np.array(returns)
        adv = returns - s_values.numpy()
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10) # output: numpy array
        return adv, returns

    def compute_targets(self, r_batch, ns_batch, d_batch):
        mean = self.actor.model(ns_batch)
        target_critic = self.critic.model([ns_batch, mean])
        y = r_batch + self.gamma * (1 - d_batch) * target_critic
        return y

    def compute_adv_bar(self, s_batch, a_batch):
        mean = self.actor.model(s_batch)
        x = tf.squeeze(a_batch) - tf.squeeze(mean)
        y = tf.squeeze(self.critic.model([s_batch, mean]))
        adv_bar = y * x         # check this
        return adv_bar

    # implements on-policy & off-policy training
    def train(self, states, actions, rewards, next_states, dones):
        n_split = len(rewards) // self.batch_size

        # convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        # IPG paper: https: // arxiv.org / abs / 1706.00387
        # Fit baseline using collected experience and compute advantages
        advantages, returns = self.compute_advantage(rewards, states, next_states, dones) # returns np.arrays

        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        # if use control variate: Compute critic-based advantages
        # and compute learning signal
        use_CV = False
        v = 0.2
        if use_CV:
            # compute critic-based advantage
            adv_bar = self.compute_adv_bar(states, actions)
            ls = advantages - adv_bar
            b = 1
        else:
            ls = advantages
            b = v

        ls *= (1 - v)

        # training batch size % n_split should be zero
        # s_split = tf.split(states, n_split)
        # a_split = tf.split(actions, n_split)
        # t_split = tf.split(returns, n_split)
        # ls_split = tf.split(ls, n_split)
        indexes = np.arange(n_split, dtype=int)

        # current policy
        mean, std = self.actor(states)
        pi = tfp.distributions.Normal(mean, std)

        a_loss_list = []
        c_loss_list = []
        np.random.shuffle(indexes)
        for _ in range(self.epochs):
            # increment global step counter
            self.time_steps += 1

            # sample a minibatch from replay buffer
            s_batch, a_batch, r_batch, ns_batch, d_batch = self.buffer.sample()

            # convert to tensors
            s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)
            a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
            r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
            ns_batch = tf.convert_to_tensor(ns_batch, dtype=tf.float32)
            d_batch = tf.convert_to_tensor(d_batch, dtype=tf.float32)

            for i in indexes:

                s_split = tf.gather(states, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                a_split = tf.gather(actions, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                ls_split = tf.gather(ls, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                t_split = tf.gather(returns, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                old_pi = pi[i * self.batch_size: (i+1) * self.batch_size]

                # update actor
                a_loss = self.actor.train(s_split, a_split, ls_split,
                                          old_pi, self.critic, b, s_batch)
                a_loss_list.append(a_loss)
                # update baseline
                v_loss = self.baseline.train(s_split, t_split)

            # update critic
            y = self.compute_targets(r_batch, ns_batch, d_batch)
            c_loss = self.critic.train(s_batch, a_batch, y)
            c_loss_list.append(c_loss)

        return np.mean(a_loss_list), np.mean(c_loss_list)

    # Validation routine
    def validate(self, max_eps=50):
        ep_reward_list = []
        for _ in range(max_eps):
            
            # empty the buffer for each episode
            if self.stack_size > 1:
                state_buffer = []

            # initial state
            state_obs = np.asarray(self.env.reset(), dtype=np.float32) / 255.0

            t = 0
            ep_reward = 0
            while True:
                if self.stack_size > 1:
                    state_buffer.append(state_obs)
                    state = self.prepare_input(state_buffer)
                else:
                    state = state_obs
                
                action = self.policy(state, deterministic=False)
                next_state_obsv, reward, done, _ = self.env.step(action)
                next_state_obsv = np.asarray(next_state_obsv, dtype=np.float32) / 255.0

                state_obs = next_state_obsv
                ep_reward += reward
                t += 1
                if done:
                    ep_reward_list.append(ep_reward)
                    break
        # outside for loop
        mean_ep_reward = np.mean(ep_reward_list)
        return mean_ep_reward

    def run(self):

        if self.filename is not None:
            self.filename = uniquify(self.path + self.filename)


        # initial state
        state_obs = np.asarray(self.env.reset(), dtype=np.float32) / 255.0

        if self.stack_size > 1:
            state_buffer = []
            next_state_buffer = []

        # record training time
        start = datetime.datetime.now()

        val_scores = []         # validation scores
        best_score = -np.inf
        s_scores = []       # All season scores
        ep_lens = []        # episode lengths 
        ep_scores = []      # episodic rewards
        self.episodes = 0   # global episode count
        for s in range(self.SEASONS):
            # discard trajectories from previous season
            states, next_states, actions, rewards, dones = [], [], [], [], []
            s_score = 0
            ep_cnt = 0      # episodes in each season
            ep_len = 0      # length of each episode
            ep_score = 0    # score of each episode
            done = False
            for _ in range(self.training_batch):    # time steps

                if self.stack_size > 1:
                    state_buffer.append(state_obs)
                    state = self.prepare_input(state_buffer)
                else:
                    state = state_obs 

                action = self.policy(state)
                next_state_obs, reward, done, _ = self.env.step(action)
                next_state_obs = np.asarray(next_state_obs, dtype=np.float32) / 255.0

                if self.stack_size > 1:
                    next_state_buffer.append(next_state_obs)
                    next_state = self.prepare_input(next_state_buffer)
                else:
                    next_state = next_state_obs

                # this is used for on-policy training
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                # store in replay buffer for off-policy training
                self.buffer.record([state, action, reward, next_state, done])

                state_obs = next_state_obs
                ep_score += reward
                ep_len += 1

                if done:
                    self.episodes += 1   # count total no of episodes
                    s_score += ep_score        # season score
                    ep_cnt += 1         # episode count in a season
                    ep_scores.append(ep_score)
                    ep_lens.append(ep_len)

                    if self.WB_LOG:
                        wandb.log({'time_steps' : self.time_steps,
                            'Episodes' : self.episodes, 
                            'mean_ep_score': np.mean(ep_scores),
                            'mean_ep_len' : np.mean(ep_lens)})
                        if self.episodes % 1000 == 0:
                            obsv_img = wandb.Image(state_obs)
                            wandb.log({'obsvn_img': obsv_img})

                    # prepare for next episode
                    state_obs = np.asarray(self.env.reset(), dtype=np.float32)
                    ep_len, ep_score = 0, 0
                    done= False

                    if self.stack_size > 1:
                        state_buffer = []
                        next_state_buffer = []

                # end of done block
            # end of one season

            # on-policy & off-policy training
            actor_loss, critic_loss = self.train(states, actions, rewards, next_states, dones)

            s_score = np.mean(ep_scores[-ep_cnt : ])
            s_scores.append(s_score)
            mean_s_score = np.mean(s_scores)
            mean_ep_len = np.mean(ep_lens)


            # validation
            val_score = self.validate()
            val_scores.append(val_score)
            mean_val_score = np.mean(val_scores)


            if mean_s_score > best_score:
                best_model_path = self.path + 'best_model/'
                self.save_model(best_model_path)
                best_score = mean_s_score
                print('Season: {}, Update best score: {}-->{}, Model saved!'.format(s, best_score, mean_s_score))
                print('Season: {}, Validation Score: {}, Mean Validation Score: {}'\
                        .format(s, val_score, mean_val_score))

            if self.WB_LOG:
                wandb.log({'Season Score' : s_score, 
                            'Mean Season Score' : mean_s_score,
                            'Actor Loss' : actor_loss,
                            'Critic Loss' :critic_loss,
                            'Mean episode length' : mean_ep_len,
                            'val_score': val_score, 
                            'mean_val_score': mean_val_score,
                            'ep_per_season': ep_cnt,
                            'Season' : s})
                
            if self.chkpt_freq is not None and s % self.chkpt_freq == 0:
                chkpt_path = self.path + 'chkpt_{}/'.format(s)
                self.save_model(chkpt_path)

            if self.filename is not None:
                with open(self.filename, 'a') as file:
                    file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                            .format(s, self.episodes, self.time_steps, mean_ep_len,
                                    s_score, mean_s_score, actor_loss, critic_loss,
                                    val_score, mean_val_score))

            if self.success_value is not None:
                if best_score > self.success_value:
                    print('Problem is solved in {} episodes with score {}'.format(self.episodes, best_score))
                    break

        # end of season-loop
        end = datetime.datetime.now()
        print('Time to Completion: {}'.format(end - start))
        print('Mean episodic score over {} episodes: {:.2f}'.format(self.episodes, np.mean(ep_scores)))
        self.env.close()

        # save the final model
        final_model_path = self.path + 'final_model/'
        self.save_model(final_model_path)
        

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        actor_file = save_path + 'ipg_actor_wts.h5'
        critic_file = save_path + 'ipg_critic_wts.h5'
        baseline_file = save_path + 'ipg_bl_wts.h5'
        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)
        self.baseline.save_weights(baseline_file)

    def load_model(self, load_path): 
        actor_file = load_path + 'ipg_actor_wts.h5'
        critic_file = load_path + 'ipg_critic_wts.h5'
        baseline_file = load_path + 'ipg_bl_wts.h5'
        self.actor.load_weights(actor_file)
        self.critic.load_weights(critic_file)
        self.baseline.load_weights(baseline_file)


###############################3
class IPGAgent2:
    def __init__(self, state_size, action_size, action_upper_bound, 
                    epochs, batch_size, buffer_capacity, 
                    lr_a, lr_c, epsilon,
                    gamma, lmbda, use_attention=False):
        self.state_size = state_size
        self.action_size = action_size
        self.upper_bound = action_upper_bound
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.use_attention = use_attention

        if len(self.state_size) == 3:
            self.image_input = True     # image input
        elif len(self.state_size) == 1:
            self.image_input = False        # vector input
        else:
            raise ValueError("Input can be a vector or an image")

        # Buffer for off-policy training
        self.buffer = Buffer(self.buffer_capacity, self.batch_size)

        if self.use_attention and self.image_input:   # attention + image input
            print('Currently Attention handles only image input')
            self.feature = AttentionFeatureNetwork(self.state_size, lr_a)
        elif self.use_attention is False and self.image_input is True:  # image input
            print('You have selected an image input')
            self.feature = FeatureNetwork(self.state_size, lr_a)
        else:       # non-image input
            print('You have selected a non-image input.')
            self.feature = None

        # create actor / critic models
        self.actor = IPGActor(self.state_size, self.action_size, self.upper_bound,
                           self.lr_a, self.epsilon, self.feature)
        self.critic = DDPGCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.baseline = Baseline(self.state_size, self.action_size, self.lr_c, self.feature)

    def policy(self, state, deterministic=False): # this needs to change
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        mean, std = self.actor(tf_state)

        if deterministic:
            action = mean
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = pi.sample()
        #action = mean + np.random.uniform(-self.upper_bound, self.upper_bound, size=mean.shape) * std
        action = tf.clip_by_value(action, -self.upper_bound, self.upper_bound)
        return action.numpy()

    def compute_advantage(self, r_batch, s_batch, ns_batch, d_batch):
        # input: tensors
        gamma = self.gamma
        lmbda = self.lmbda
        s_values = tf.squeeze(self.baseline.model(s_batch)) # input: tensor
        ns_values = tf.squeeze(self.baseline.model(ns_batch))
        returns = []
        gae = 0     # generalized advantage estimate
        for i in reversed(range(len(r_batch))):
            delta = r_batch[i] + gamma * ns_values[i] * (1 - d_batch[i]) - s_values[i]
            gae = delta + gamma * lmbda * (1 - d_batch[i]) * gae
            returns. insert(0, gae + s_values[i])

        returns = np.array(returns)
        adv = returns - s_values.numpy()
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10) # output: numpy array
        return adv, returns

    def compute_targets(self, r_batch, ns_batch, d_batch):
        mean = self.actor.model(ns_batch)
        target_critic = self.critic.model([ns_batch, mean])
        y = r_batch + self.gamma * (1 - d_batch) * target_critic
        return y

    def compute_adv_bar(self, s_batch, a_batch):
        mean = self.actor.model(s_batch)
        x = tf.squeeze(a_batch) - tf.squeeze(mean)
        y = tf.squeeze(self.critic.model([s_batch, mean]))
        adv_bar = y * x         # check this
        return adv_bar

    # implements on-policy & off-policy training
    def replay(self, states, actions, rewards, next_states, dones):
        n_split = len(rewards) // self.batch_size

        # convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        # IPG paper: https: // arxiv.org / abs / 1706.00387
        # Fit baseline using collected experience and compute advantages
        advantages, returns = self.compute_advantage(rewards, states, next_states, dones) # returns np.arrays

        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        # if use control variate: Compute critic-based advantages
        # and compute learning signal
        use_CV = False
        v = 0.2
        if use_CV:
            # compute critic-based advantage
            adv_bar = self.compute_adv_bar(states, actions)
            ls = advantages - adv_bar
            b = 1
        else:
            ls = advantages
            b = v

        ls *= (1 - v)

        # training batch size % n_split should be zero
        # s_split = tf.split(states, n_split)
        # a_split = tf.split(actions, n_split)
        # t_split = tf.split(returns, n_split)
        # ls_split = tf.split(ls, n_split)
        indexes = np.arange(n_split, dtype=int)

        # current policy
        mean, std = self.actor(states)
        pi = tfp.distributions.Normal(mean, std)

        a_loss_list = []
        c_loss_list = []
        np.random.shuffle(indexes)
        for _ in range(self.epochs):
            s_batch, a_batch, r_batch, ns_batch, d_batch = self.buffer.sample()

            # convert to tensors
            s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)
            a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
            r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
            ns_batch = tf.convert_to_tensor(ns_batch, dtype=tf.float32)
            d_batch = tf.convert_to_tensor(d_batch, dtype=tf.float32)

            for i in indexes:

                s_split = tf.gather(states, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                a_split = tf.gather(actions, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                ls_split = tf.gather(ls, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                t_split = tf.gather(returns, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                old_pi = pi[i * self.batch_size: (i+1) * self.batch_size]

                # update actor
                a_loss = self.actor.train(s_split, a_split, ls_split,
                                          old_pi, self.critic, b, s_batch)
                a_loss_list.append(a_loss)
                # update baseline
                v_loss = self.baseline.train(s_split, t_split)

            # update critic
            y = self.compute_targets(r_batch, ns_batch, d_batch)
            c_loss = self.critic.train(s_batch, a_batch, y)
            c_loss_list.append(c_loss)

        return np.mean(a_loss_list), np.mean(c_loss_list)

    def save_model(self, save_path):
        actor_file = save_path + 'ipg_actor_wts.h5'
        critic_file = save_path + 'ipg_critic_wts.h5'
        baseline_file = save_path + 'ipg_bl_wts.h5'
        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)
        self.baseline.save_weights(baseline_file)

    def load_model(self, load_path):
        actor_file = load_path + 'ipg_actor_wts.h5' 
        critic_file = load_path + 'ipg_critic_wts.h5'
        baseline_file = load_path + 'ipg_bl_wts.h5'
        self.actor.load_weights(actor_file)
        self.critic.load_weights(critic_file)
        self.baseline.load_weights(baseline_file)






