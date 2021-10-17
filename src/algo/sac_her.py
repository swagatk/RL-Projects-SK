"""
SAC + HER Algorithm.

We implement two strategies for selecting hind_goal

- Last state as hind goal
- Last successful state

"""
from inspect import currentframe
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import dtype
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
from common.FeatureNet import FeatureNetwork
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
        self.model = self._build_net()    # returns mean action
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        state_input = tf.keras.layers.Input(shape=self.state_size)
        goal_input = tf.keras.layers.Input(shape=self.goal_size)

        if self.feature_model is None:
            f_state = tf.keras.layers.Dense(128, activation="relu")(state_input)
            f_goal = tf.keras.layers.Dense(128, activation="relu")(goal_input)
        else:
            f_state = self.feature_model(state_input)
            f_goal = self.feature_model(goal_input)

        f = tf.keras.layers.Concatenate()([f_state, f_goal])
        f = tf.keras.layers.Dense(128, activation='relu')(f)
        f = tf.keras.layers.Dense(64, activation="relu")(f)
        mu = tf.keras.layers.Dense(self.action_size[0], activation='tanh')(f)
        mu = tf.math.multiply(mu, self.upper_bound)
        log_sig = tf.keras.layers.Dense(self.action_size[0])(f)
        model = tf.keras.Model(inputs=[state_input, goal_input], outputs=[mu, log_sig], name='actor')
        model.summary()
        tf.keras.utils.plot_model(model, to_file='sac_her_actor_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state, goal):
        # input: tensor, output: tensor
        mean, log_sigma = self.model([state, goal])
        std = tf.exp(log_sigma)
        return mean, std 

    def policy(self, state, goal):
        # input: tensor;  output: tensor
        mean, std = self.__call__(state, goal)

        pi = tfp.distributions.Normal(mean, std)
        action_ = pi.sample()
        log_pi_ = pi.log_prob(action_)
        action = tf.clip_by_value(action_, -self.upper_bound, self.upper_bound)
        log_pi_a =  tf.reduce_sum((log_pi_ - tf.math.log( 1 - action ** 2 + 1e-16)), axis=-1, keepdims=True)

        # if tf.rank(action) < 1:     # scalar
        #     log_pi_a = log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)
        # elif 1 <= tf.rank(action) < 2:  # vector
        #     log_pi_a = tf.reduce_sum((log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)), axis=0, keepdims=True)
        # else:   # matrix
        #     log_pi_a = tf.reduce_sum((log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)), axis=1, keepdims=True)

        return action, log_pi_a
    
    def train(self, states, alpha, critic1, critic2, goals):
        with tf.GradientTape() as tape:
            # obtain actions using the current policy
            actions, log_pi_a = self.policy(states, goals)
            
            # estimate q values 
            q1 = critic1(states, goals, actions)
            q2 = critic2(states, goals, actions)

            # Apply the clipped double Q trick
            # get the minimum of two Q values
            min_q = tf.minimum(q1, q2)

            # compute soft q target using entropy
            soft_q = min_q - alpha * log_pi_a
            actor_loss = -tf.reduce_mean(soft_q)
            actor_wts = self.model.trainable_variables
        # outside gradient tape block
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
        self.goal_size = state_size
        self.lr = learning_rate
        self.gamma = gamma          # discount factor
        

        # create NN model
        self.feature = feature_model
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        state_input = layers.Input(shape=self.state_size)
        goal_input = layers.Input(shape=self.goal_size)

        if self.feature is None:
            f_state = layers.Dense(128, activation='relu')(state_input)
            f_goal = layers.Dense(128, activation='relu')(goal_input)
        else:
            f_state = self.feature(state_input)
            f_goal = self.feature(goal_input)

        f = layers.Concatenate()([f_state, f_goal])

        state_out = layers.Dense(32, activation='relu')(f)
        state_out = layers.Dense(32, activation='relu')(state_out)

        # action as input
        action_input = layers.Input(shape=self.action_size)
        action_out = layers.Dense(32, activation='relu')(action_input)

        concat = layers.Concatenate()([state_out, action_out])
        out = layers.Dense(128, activation='relu')(concat)
        out = layers.Dense(64, activation='relu')(out)
        q_value = layers.Dense(1)(out)
        model = tf.keras.Model(inputs=[state_input, goal_input, action_input], outputs=q_value)
        model.summary()
        tf.keras.utils.plot_model(model, to_file='sac_critic_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state, goal, action):
        # input: tensors
        q_value = tf.squeeze(self.model([state, goal, action]))
        return q_value

    def train(self, states, actions, rewards, next_states, dones, actor,
              target_critic1, target_critic2, alpha, goals):
        with tf.GradientTape() as tape:
            # Get Q estimates using actions from replay buffer
            q_values = self.__call__(states, goals, actions) 

            # Sample actions from the policy network for next states
            a_next, log_pi_a = actor.policy(next_states, goals)

            # Get Q value estimates from target Q networks
            q1_target = target_critic1(next_states, goals, a_next)
            q2_target = target_critic2(next_states, goals, a_next)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - alpha * log_pi_a

            y = tf.stop_gradient(rewards + self.gamma * (1 - dones) * soft_q_target)
            critic_loss = tf.math.reduce_mean(tf.square(y - q_values))
            critic_wts = self.model.trainable_variables
        # outside gradient tape
        critic_grad = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grad, critic_wts))
        return critic_loss.numpy()
    
    def train2(self, state_batch, action_batch, goal_batch, y):
        with tf.GradientTape() as tape:
            q_values = self.__call__(state_batch, goal_batch, action_batch)
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


class SACHERAgent:
    def __init__(self, state_size, action_size, upper_bound,
                buffer_capacity=100000, batch_size=128, epochs=50,
                learning_rate=0.0003, gamma=0.99, polyak=0.995, 
                alpha=0.2, use_attention=None, use_her=None ):
        self.action_size = action_size
        self.state_size = state_size
        self.goal_size = state_size
        self.upper_bound = upper_bound 
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_capacity =buffer_capacity
        self.target_entropy = -tf.constant(np.prod(self.action_size), dtype=tf.float32)
        self.gamma = gamma                  # discount factor
        self.polyak = polyak                      # polyak averaging factor
        self.use_attention = use_attention
        self.use_her = use_her
        self.episodes = 0

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


        # Create an actor network
        self.actor = SACHERActor(self.state_size, self.action_size, self.upper_bound,
                              self.lr, self.feature)

        # create two critics
        self.critic1 = SACHERCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.feature)
        self.critic2 = SACHERCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.feature)

        # create two target critics
        self.target_critic1 = SACHERCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.feature)
        self.target_critic2 = SACHERCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.feature)

        # create alpha as a trainable variable
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(self.lr)

        # Buffer for off-policy training
        self.buffer = HERBuffer(self.buffer_capacity, self.batch_size)

    def sample_action(self, state, goal):
        # input: numpy array
        # output: numpy array
        if state.ndim < len(self.state_size) + 1:      # single sample
            tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            tf_goal = tf.expand_dims(tf.convert_to_tensor(goal, dtype=tf.float32), 0)
        else:       # for a batch of samples
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
            tf_goal = tf.convert_to_tensor(goal, dtype=tf.float32)

        action, log_pi = self.actor.policy(tf_state, tf_goal) # returns tensors
        return action[0].numpy(), log_pi[0].numpy()

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
        for theta_target, theta in zip(self.target_critic1.model.trainable_variables,
                    self.critic1.model.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta
        
        for theta_target, theta in zip(self.target_critic2.model.trainable_variables,
                    self.critic2.model.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta

    def update_q_networks(self, states, actions, rewards, next_states, dones, goals):
        # input: tensors, output: numpy arrays

        # sample actions from current policy for next states
        pi_a, log_pi_a = self.actor.policy(next_states, goals)

        # get Q value estimates from target Q networks
        q1_target = self.target_critic1(next_states, goals, pi_a)
        q2_target = self.critic2(next_states, goals, pi_a)

        # apply the clipped double Q trick
        min_q_target = tf.minimum(q1_target, q2_target)

        # add the entropy term to get soft Q target
        soft_q_target = min_q_target  - self.alpha * log_pi_a  

        # target signal for critic network
        y = rewards + self.gamma * (1 - dones) * soft_q_target

        c1_loss = self.critic1.train2(states, goals, actions, y)
        c2_loss = self.critic2.train2(states, goals, actions, y)

        mean_c_loss = np.mean([c1_loss, c2_loss])
        return mean_c_loss 

    def train(self, CRIT_T2=True):

        critic_losses, actor_losses, alpha_losses = [], [], []
        for _ in range(self.epochs):

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
            goal = np.asarray(env.reset(), dtype=np.float32) / 255.0

            t = 0
            ep_reward = 0
            while True:
                action, _ = self.policy(state, goal)
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

    def run(self, env, seasons=50, training_batch=1024, train_freq=20,
                chkpt_freq=None, success_value=None, filename=None,
                wb_log=False, path='./'):

        if filename is not None: 
            filename = uniquify(path + filename)


        # initial state
        state = np.asarray(env.reset(), dtype=np.float32) / 255.0
        goal = np.asarray(env.reset(), dtype=np.float32) / 255.0

        if self.her_strategy == 'success': # Store the successful states   
            desired_goals = deque(maxlen=1000)

        start = datetime.datetime.now()
        val_scores = []                 # validation scores 
        best_score = -np.inf
        ep_lens = []        # episodic length
        ep_scores = []      # All episodic scores
        s_scores = []       # season scores
        self.episodes = 0       # total episode count
        ep_actor_losses = []    # actor losses
        ep_critic_losses = []   # critic losses

        for s in range(seasons):
            ep_experience = []    # required for HER
            s_score = 0         # season score 
            ep_len = 0          # episode length
            ep_score = 0        # score for each episode
            ep_cnt = 0          # no. of episodes in each season
            done = False
            for t in range(training_batch):
                action, _ = self.policy(state, goal)
                next_state, reward, done, _ = env.step(action)
                next_state = np.asarray(next_state, dtype=np.float32) / 255.0

                if self.her_strategy == 'success':
                    if reward == 1:     # store successful states
                        desired_goals.append([state, action, reward, next_state, done, goal])            

                # store in replay buffer for off-policy training
                self.buffer.record([state, action, reward, next_state, done, goal])

                # also store experience in temporary buffer
                ep_experience.append([state, action, reward, next_state, done, goal])

                state = next_state
                ep_score += reward
                ep_len += 1     # no. of time steps in each episode
                
                if done:
                    s_score += ep_score
                    ep_cnt += 1     # no. of episodes in each season
                    self.episodes += 1  # total episode count
                    ep_scores.append(ep_score)
                    ep_lens.append(ep_len) 

                    # HER strategies
                    if self.her_strategy == 'final':    # final state strategy
                        hind_goal = ep_experience[-1][3]      
                    elif self.her_strategy == 'success':    # successful states
                        if len(desired_goals) < 1:
                            hind_goal = ep_experience[-1][3]      
                        else:
                            index = np.random.choice(len(desired_goals))
                            hind_goal = desired_goals[index][3]    
                    elif self.her_strategy == 'future': # future state strategy
                        hind_goal = None
                    else:
                        raise ValueError('Invalid choice for HER strategy. Exiting ..')

                    self.add_her_experience(ep_experience, hind_goal, extract_feature=False)
                    ep_experience = [] # clear temporary buffer

                    # off-policy training after each episode
                    if self.episodes % train_freq == 0:
                        a_loss, c_loss, alpha_loss = self.train()

                        ep_actor_losses.append(a_loss)
                        ep_critic_losses.append(c_loss)

                        if wb_log: 
                            wandb.log({
                            'ep_actor_loss' : a_loss,
                            'ep_critic_loss' : c_loss,
                            'ep_alpha_loss' : alpha_loss})

                    if wb_log: 
                        wandb.log({
                            'Episodes' : self.episodes, 
                            'mean_ep_score': np.mean(ep_scores),
                            'mean_ep_len' : np.mean(ep_lens)})
                    
                    # prepare for next episode
                    state = np.asarray(env.reset(), dtype=np.float32) / 255.0
                    goal = np.asarray(env.reset(), dtype=np.float32) / 255.0
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

            # run validation once in each iteration
            val_score = self.validate()
            val_scores.append(val_score)
            mean_val_score = np.mean(val_scores)

            if mean_s_score > best_score:
                best_model_path = self.path + 'best_model/'
                os.makedirs(best_model_path, exist_ok=True)
                self.save_model(best_model_path)
                best_score = mean_s_score
                print('Season: {}, Update best score: {}-->{}, Model saved!'.format(s, best_score, mean_ep_score))
                print('Season: {}, Validation Score: {}, Mean Validation Score: {}' \
                .format(s, val_score, mean_val_score))

            if wb_log:
                wandb.log({'Season Score' : s_score, 
                            'Mean Season Score' : mean_s_score,
                            'Actor Loss' : mean_actor_loss,
                            'Critic Loss' : mean_critic_loss,
                            'Mean episode length' : mean_ep_len,
                            'val_score': val_score, 
                            'mean_val_score': mean_val_score,
                            'ep_per_season' : ep_cnt,
                            'Season' : s})

            if chkpt_freq is not None and s % self.chkpt_freq == 0:
                chkpt_path = self.path + 'chkpt_{}/'.format(s)
                os.makedirs(chkpt_path, exist_ok=True)
                self.save_model(chkpt_path)

            if filename is not None:
                with open(filename, 'a') as file:
                    file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                            .format(s, self.episodes, ep_cnt, mean_ep_len,
                                    s_score, mean_s_score, mean_actor_loss, mean_critic_loss, alpha_loss,
                                    val_score, mean_val_score))

            if success_value is not None:
                if best_score > self.success_value:
                    print('Problem is solved in {} episodes with score {}'.format(s, best_score))
                    print('Mean Episodic score: {}'.format(mean_s_score))
                    break

        # end of season-loop
        end = datetime.datetime.now()
        print('Time to Completion: {}'.format(end - start))
        env.close()
        print('Mean episodic score over {} episodes: {:.2f}'.format(self.episodes, np.mean(ep_scores)))

    def her_reward_func_1(self, state, goal, thr=0.3):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        tf_goal = tf.expand_dims(tf.convert_to_tensor(goal, dtype=tf.float32), axis=0)
        
        state_feature = tf.squeeze(self.feature(tf_state))
        goal_feature = tf.squeeze(self.feature(tf_goal))

        good_done = tf.linalg.norm(state_feature - goal_feature) <= thr
        reward = 1 if good_done else 0
        return good_done, reward

    def her_reward_func_2(self, state, goal, thr=0.2):
        # input: numpy array, output: numpy value
        good_done = np.linalg.norm(state - goal) <= thr 
        reward = 1 if good_done else 0
        return good_done, reward

    def add_her_experience(self, ep_experience, hind_goal, extract_feature=False):
        for i in range(len(ep_experience)):
            if hind_goal is None:   # future state strategy
                future = np.random.randint(i, len(ep_experience))
                goal_ = ep_experience[future][3]
            else:
                goal_ = hind_goal

            state_ = ep_experience[i][0]
            action_ = ep_experience[i][1]
            next_state_ = ep_experience[i][3]

            if extract_feature:     
                done_, reward_ = self.her_reward_func_1(next_state_, goal_)
            else:
                done_, reward_ = self.her_reward_func_2(next_state_, goal_)

            # add new experience to the main buffer
            self.buffer.record([state_, action_, reward_, next_state_, done_, goal_])

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

