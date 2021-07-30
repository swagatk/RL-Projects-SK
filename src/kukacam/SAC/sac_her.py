"""
SAC + HER Algorithm.
"""
from inspect import currentframe
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
# add current directory to python module path
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)
sys.path.append(os.path.dirname(current_directory))

# Local imports
from common.FeatureNet import FeatureNetwork, AttentionFeatureNetwork
from common.buffer import HERBuffer
from common.utils import uniquify


###############
# ACTOR NETWORK
###############

class SACHERActor:
    '''
    goals and states have same size
    '''
    def __init__(self, state_size, action_size, upper_bound,
                 learning_rate, feature):
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.upper_bound = upper_bound
        self.goal_size = state_size
        
        # create NN models
        self.feature_model = feature
        self.model = self._build_net(trainable=True)    # returns mean action
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # additional output
        logstd = tf.Variable(np.zeros(shape=self.action_size, dtype=np.float32))
        self.model.logstd = logstd
        self.model.trainable_variables.append(logstd)

    def _build_net(self, trainable=True):
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)
        state_input = tf.keras.layers.Input(shape=self.state_size)
        goal_input = tf.keras.layers.Input(shape=self.goal_size)

        if self.feature_model is None:
            f_state = tf.keras.layers.Dense(128, activation="relu", trainable=trainable)(state_input)
            f_goal = tf.keras.layers.Dense(128, activation="relu", trainable=trainable)(goal_input)
        else:
            f_state = self.feature_model(state_input)
            f_goal = self.feature_model(goal_input)

        f = tf.keras.layers.Concatenate()([f_state, f_goal])
        f = tf.keras.layers.Dense(128, activation='relu', trainable=trainable)(f)
        f = tf.keras.layers.Dense(64, activation="relu", trainable=trainable)(f)
        net_out = tf.keras.layers.Dense(self.action_size[0], activation='tanh',
                                        kernel_initializer=last_init, trainable=trainable)(f)
        net_out = net_out * self.upper_bound  # element-wise product
        model = tf.keras.Model(inputs=[state_input, goal_input], outputs=net_out, name='actor')
        model.summary()
        tf.keras.utils.plot_model(model, to_file='sac_her_actor_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state, goal):
        # input: tensor
        # output: tensor
        mean = tf.squeeze(self.model([state, goal]))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std 

    def policy(self, state, goal):
        # input: tensor
        # output: tensor
        mean = tf.squeeze(self.model([state, goal]))
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
    
    def train(self, states, alpha, critic1, critic2, goals):
        with tf.GradientTape() as tape:
            # obtain actions using the current policy
            actions, log_pi_a = self.policy(states, goals)
            
            # estimate q values 
            q1 = critic1(states, actions)
            q2 = critic2(states, actions)

            # Apply the clipped double Q trick
            # get the minimum of two Q values
            min_q = tf.minimum(q1, q2)

            # compute soft q target using entropy
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


class SACHERCritic:
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
        tf.keras.utils.plot_model(model, to_file='sac_critic_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state, action):
        # input: tensors
        q_value = tf.squeeze(self.model([state, action]))
        return q_value

    def train(self, states, actions, rewards, next_states, dones, actor,
              target_critic1, target_critic2, alpha, goals):
        with tf.GradientTape() as tape:
            # Get Q estimates using actions from replay buffer
            q_values = tf.squeeze(self.model([states, actions]))

            # Sample actions from the policy network for next states
            a_next, log_pi_a = actor.policy(next_states, goals)

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
        # outside gradient tape
        critic_wts = self.model.trainable_variables
        critic_grad = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grad, critic_wts))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class SACHERAgent:
    def __init__(self, env, success_value, epochs,
                 training_batch, batch_size, buffer_capacity, lr_a=0.0003, lr_c=0.0003,
                 gamma=0.99, tau=0.995, alpha=0.2, use_attention=False, 
                 filename=None, wb_log=False, validation=True, chkpt=False, save_path='./'):
        self.env = env
        self.action_size = self.env.action_space.shape
        self.state_size = self.env.observation_space.shape
        self.upper_bound = np.squeeze(self.env.action_space.high)

        self.episodes = 0               # total episode count
        self.time_steps = 0             # total time steps
        self.success_value = success_value
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.epochs = epochs
        self.training_batch = training_batch    # no. of time steps in each season
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_entropy = -tf.constant(np.prod(self.action_size), dtype=tf.float32)
        self.gamma = gamma                  # discount factor
        self.tau = tau                      # polyak averaging factor
        self.use_attention = use_attention
        self.filename = filename
        self.WB_LOG = wb_log
        self.validation = validation 
        self.path = save_path
        self.chkpt = chkpt                  # store checkpoints

        if len(self.state_size) == 3:
            self.image_input = True     # image input
        elif len(self.state_size) == 1:
            self.image_input = False        # vector input
        else:
            raise ValueError("Input can be a vector or an image")

        # Select a suitable feature extractor
        if self.use_attention and self.image_input:   # attention + image input
            print('Currently Attention handles only image input')
            self.feature = AttentionFeatureNetwork(self.state_size, lr_a)
        elif self.use_attention is False and self.image_input is True:  # image input
            print('You have selected an image input')
            self.feature = FeatureNetwork(self.state_size, lr_a)
        else:       # non-image input
            print('You have selected a non-image input.')
            self.feature = None

        self.actor = SACHERActor(self.state_size, self.action_size, self.upper_bound,
                              self.lr_a, self.feature)

        # create two critics
        self.critic1 = SACHERCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.critic2 = SACHERCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)

        # create two target critics
        self.target_critic1 = SACHERCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.target_critic2 = SACHERCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)

        # create alpha as a trainable variable
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr_a)

        # Buffer for off-policy training
        self.buffer = HERBuffer(self.buffer_capacity, self.batch_size)

    def policy(self, state, goal):
        # input: numpy array
        # output: numpy array
        if state.ndim < len(self.state_size) + 1:      # single sample
            tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            tf_goal = tf.expand_dims(tf.convert_to_tensor(goal, dtype=tf.float32), 0)
        else:       # for a batch of samples
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
            tf_goal = tf.convert_to_tensor(goal, dtype=tf.float32)

        action, log_pi = self.actor.policy(tf_state, tf_goal) # returns tensors
        return action.numpy(), log_pi.numpy()

    def update_alpha(self, states, goals):
        # input: tensor
        with tf.GradientTape() as tape:
            # sample actions from the policy for the current states
            _, log_pi_a = self.actor.policy(states, goals)
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

    def update_q_networks(self, states, actions, rewards, next_states, dones, goals):
        # input: tensors, output: numpy arrays

        # sample actions from current policy for next states
        pi_a, log_pi_a = self.actor.policy(next_states, goals)

        # get Q value estimates from target Q networks
        q1_target = self.target_critic1(next_states, pi_a)
        q2_target = self.critic2(next_states, pi_a)

        # apply the clipped double Q trick
        min_q_target = tf.minimum(q1_target, q2_target)

        # add the entropy term to get soft Q target
        soft_q_target = min_q_target  - self.alpha * log_pi_a  

        # target signal for critic network
        y = rewards + self.gamma * (1 - dones) * soft_q_target

        c1_loss = self.critic1.train2(states, actions, y)
        c2_loss = self.critic2.train2(states, actions, y)

        mean_c_loss = np.mean([c1_loss, c2_loss])
        return mean_c_loss 

    def train(self, CRIT_T2=True):
        critic_losses, actor_losses, alpha_losses = [], [], []
        for epoch in range(self.epochs):

            # increment global step counter
            self.time_steps += 1

            # sample a minibatch from the replay buffer
            states, actions, rewards, next_states, dones, goals = self.buffer.sample()

            # convert to tensors
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(np.array(dones, dtype=np.float32), dtype=tf.float32)
            goals = tf.convert_to_tensor(goals, dtype=tf.float32)

            # update Q (Critic) network weights
            if CRIT_T2 is not True:
                c1_loss = self.critic1.train(states, actions, rewards, next_states, dones,
                                            self.actor, self.target_critic1, self.target_critic2,
                                            self.alpha, goals)

                c2_loss = self.critic2.train(states, actions, rewards, next_states, dones,
                                            self.actor, self.target_critic1, self.target_critic2,
                                            self.alpha, goals)
                critic_loss = np.mean([c1_loss, c2_loss])
            else:
                critic_loss = self.update_q_networks(states, actions, rewards, next_states, dones, goals)

            # update actor
            actor_loss = self.actor.train(states, self.alpha, self.critic1, self.critic2, goals)

            # update alpha - entropy coefficient
            alpha_loss = self.update_alpha(states, goals)

            # update target network weights
            self.update_target_networks()

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)
        # epoch loop ends here
        mean_critic_loss = np.mean(critic_loss)
        mean_actor_loss = np.mean(actor_losses)
        mean_alpha_loss = np.mean(alpha_losses)
        return  mean_actor_loss, mean_critic_loss, mean_alpha_loss

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

    def run(self):

        if self.filename is not None: 
            self.filename = uniquify(self.path + self.filename)

        if self.validation:
            val_scores = deque(maxlen=50)
            val_score = 0

        # initial state
        state = np.asarray(self.env.reset(), dtype=np.float32) / 255.0
        goal = np.asarray(self.env.reset(), dtype=np.float32) / 255.0

        start = datetime.datetime.now()
        best_score = -np.inf
        ep_lens = []        # episodic length
        ep_scores = []      # All episodic scores
        s_scores = []       # season scores
        self.episodes = 0       # total episode count
        ep_actor_losses = []    # actor losses
        ep_critic_losses = []   # critic losses

        for s in range(self.seasons):
            temp_experience = []    # required for HER
            s_score = 0         # season score 
            ep_len = 0          # episode length
            ep_score = 0        # score for each episode
            ep_cnt = 0          # no. of episodes in each season
            done = False
            for t in range(self.training_batch):
                action, _ = self.policy(state, goal)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.asarray(next_state, dtype=np.float32) / 255.0

                # store in replay buffer for off-policy training
                self.buffer.record([state, action, reward, next_state, done, goal])

                # also store experience in temporary buffer
                temp_experience.append([state, action, reward, next_state, done, goal])

                state = next_state
                ep_score += reward
                ep_len += 1     # no. of time steps in each episode
                
                if done:
                    s_score += ep_score
                    ep_cnt += 1     # no. of episodes in each season
                    self.episodes += 1  # total episode count
                    ep_scores.append(ep_score)
                    ep_lens.append(ep_len) 

                    # HER: Final state strategy
                    hind_goal = temp_experience[-1][3]
                    self.add_her_experience(temp_experience, hind_goal)
                    temp_experience = [] # clear temporary buffer

                    # off-policy training after each episode
                    a_loss, c_loss, alpha_loss = self.train()

                    ep_actor_losses.append(a_loss)
                    ep_critic_losses.append(c_loss)

                    if self.WB_LOG:
                        wandb.log({'time_steps' : self.time_steps,
                            'Episodes' : self.episodes, 
                            'mean_ep_score': np.mean(ep_scores),
                            'ep_actor_loss' : a_loss,
                            'ep_critic_loss' : c_loss,
                            'ep_alpha_loss' : alpha_loss,
                            'mean_ep_len' : np.mean(ep_lens)})
                    
                    # prepare for next episode
                    state = np.asarray(self.env.reset(), dtype=np.float32) / 255.0
                    goal = np.asarray(self.env.reset(), dtype=np.float32) / 255.0
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
                print('Episode: {}, Update best score: {}-->{}, Model saved!'.format(ep, best_score, mean_ep_score))
                best_score = mean_s_score

            if self.validation:
                val_score = self.validate(self.env)
                val_scores.append(val_score)
                mean_val_score = np.mean(val_scores)
                print('Season: {}, Validation Score: {}, Mean Validation Score: {}' \
                        .format(s, val_score, mean_val_score))
                if self.WB_LOG:
                    wandb.log({'val_score': val_score, 
                                'mean_val_score': val_score})

            if self.WB_LOG:
                wandb.log({'Season Score' : s_score, 
                            'Mean Season Score' : mean_s_score,
                            'Actor Loss' : mean_actor_loss,
                            'Critic Loss' : mean_critic_loss,
                            'Mean episode length' : mean_ep_len,
                            'Season' : s})

            if self.chkpt:
                chkpt_path = self.path + 'chkpt/'
                os.makedirs(chkpt_path, exist_ok=True)
                self.save_model(chkpt_path)

            if self.filename is not None:
                with open(self.filename, 'a') as file:
                    file.write('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                            .format(s, self.time_steps, self.episodes, mean_ep_len,
                                    s_score, mean_s_score, mean_actor_loss, mean_critic_loss, alpha_loss))

            if self.success_value is not None:
                if best_score > self.success_value:
                    print('Problem is solved in {} episodes with score {}'.format(s, best_score))
                    print('Mean Episodic score: {}'.format(mean_s_score))
                    break

        # end of season-loop
        end = datetime.datetime.now()
        print('Time to Completion: {}'.format(end - start))
        self.env.close()
        print('Mean episodic score over {} episodes: {:.2f}'.format(self.episodes, np.mean(ep_scores)))

    def her_reward_func(self, state, goal, thr=0.3):
        good_done = np.linalg.norm(state - goal) <= thr
        reward = 1 if good_done else 0
        return good_done, reward

    def add_her_experience(self, ep_experience, hind_goal):
        for i in range(len(ep_experience)):
            if hind_goal is None:
                future = np.random.randint(i, len(ep_experience))
                goal_ = ep_experience[future][3]
            else:
                goal_ = hind_goal
            state_ = ep_experience[i][0]
            action_ = ep_experience[i][1]
            next_state_ = ep_experience[i][3]
            current_goal = ep_experience[i][5]
            done_, reward_ = self.her_reward_func(next_state_, goal_)
            # add new experience to the main buffer
            self.buffer.record([state_, action_, reward_, next_state_, done_, goal_])

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


#####################
class SACHERAgent2:
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
        self.goal_size = self.state_size
        self.thr = 0.3                      # threshold required for her_reward_function

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
        self.actor = SACHERActor(self.state_size, self.action_size, self.upper_bound,
                              self.lr_a, self.feature)

        # create two critics
        self.critic1 = SACHERCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.critic2 = SACHERCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)

        # create two target critics
        self.target_critic1 = SACHERCritic(self.state_size, self.action_size,
                                        self.lr_c, self.gamma, self.feature)
        self.target_critic2 = SACHERCritic(self.state_size, self.action_size,
                                        self.lr_c, self.gamma, self.feature)

        # create alpha as a trainable variable
        # This is the entropy coefficient required for soft target
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(self.lr_a)

        # Buffer for off-policy training
        self.buffer = HERBuffer(self.buffer_capacity, self.batch_size)

    def policy(self, state, goal):
        # input: numpy array
        # output: numpy array
        if state.ndim < len(self.state_size) + 1:      # single sample
            tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            tf_goal = tf.expand_dims(tf.convert_to_tensor(goal, dtype=tf.float32), 0)
        else:       # for a batch of samples
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
            tf_goal = tf.convert_to_tensor(goal, dtype=tf.float32)

        action, log_pi = self.actor.policy(tf_state, tf_goal) # returns tensors
        return action.numpy(), log_pi.numpy()

    def update_alpha(self, states, goals):
        # input: tensor
        with tf.GradientTape() as tape:
            # sample actions from the policy for the current states
            _, log_pi_a = self.actor.policy(states, goals)
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

    def update_q_networks(self, states, actions, rewards, next_states, dones, goals):
        # input: tensors, output: numpy arrays

        # sample actions from current policy for next states
        pi_a, log_pi_a = self.actor.policy(next_states, goals)

        # get Q value estimates from target Q networks
        q1_target = self.target_critic1(next_states, pi_a)
        q2_target = self.critic2(next_states, pi_a)

        # apply the clipped double Q trick
        min_q_target = tf.minimum(q1_target, q2_target)

        # add the entropy term to get soft Q target
        soft_q_target = min_q_target  - self.alpha * log_pi_a  

        # target signal for critic network
        y = rewards + self.gamma * (1 - dones) * soft_q_target

        c1_loss = self.critic1.train2(states, actions, y)
        c2_loss = self.critic2.train2(states, actions, y)

        mean_c_loss = np.mean([c1_loss, c2_loss])
        return mean_c_loss 

    def train(self, CRIT_T2=True):
        critic_losses, actor_losses, alpha_losses = [], [], []
        for epoch in range(self.epochs):
            # sample a minibatch from the replay buffer
            states, actions, rewards, next_states, dones, goals = self.buffer.sample()

            # convert to tensors
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(np.array(dones, dtype=np.float32), dtype=tf.float32)
            goals = tf.convert_to_tensor(goals, dtype=tf.float32)

            # update Q (Critic) network weights
            if CRIT_T2 is not True:
                c1_loss = self.critic1.train(states, actions, rewards, next_states, dones,
                                            self.actor, self.target_critic1, self.target_critic2,
                                            self.alpha, goals)

                c2_loss = self.critic2.train(states, actions, rewards, next_states, dones,
                                            self.actor, self.target_critic1, self.target_critic2,
                                            self.alpha, goals)
                critic_loss = np.mean([c1_loss, c2_loss])
            else:
                critic_loss = self.update_q_networks(states, actions, rewards, next_states, dones, goals)

            # update actor
            actor_loss = self.actor.train(states, self.alpha, self.critic1, self.critic2, goals)

            # update alpha - entropy coefficient
            alpha_loss = self.update_alpha(states, goals)

            # update target network weights
            self.update_target_networks()

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)
        # epoch loop ends here
        mean_critic_loss = np.mean(critic_loss)
        mean_actor_loss = np.mean(actor_losses)
        mean_alpha_loss = np.mean(alpha_losses)

        return  mean_actor_loss, mean_critic_loss, mean_alpha_loss

    def her_reward_func(self, state, goal, thr=0.3):
        good_done = np.linalg.norm(state - goal) <= thr
        reward = 1 if good_done else 0
        return good_done, reward

    def add_her_experience(self, ep_experience, hind_goal):
        for i in range(len(ep_experience)):
            if hind_goal is None:
                future = np.random.randint(i, len(ep_experience))
                goal_ = ep_experience[future][3]
            else:
                goal_ = hind_goal
            state_ = ep_experience[i][0]
            action_ = ep_experience[i][1]
            next_state_ = ep_experience[i][3]
            current_goal = ep_experience[i][5]
            done_, reward_ = self.her_reward_func(next_state_, goal_)
            # add new experience to the main buffer
            self.buffer.record([state_, action_, reward_, next_state_, done_, goal_])

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



