"""
DDPG Algorithm for pendulum
"""

import sys
import os 
import gym
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from utils.networks import get_session
from utils.stats import gather_stats 
#from utils.memory_buffer import MemoryBuffer 
import keras.backend as K
from keras.backend.tensorflow_backend import set_session 
import tensorflow as tf
from keras.initializers import RandomUniform
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, concatenate
from keras.layers import BatchNormalization, GaussianNoise, Flatten
from keras.utils import plot_model
import pdb # for debugging
from collections import deque


####################################################
## Actor
#####################
class Actor:
    def __init__(self, inp_dim, out_dim, act_range, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.model = self._build_net()
        self.target_model = self._build_net()

    def _build_net(self):
        inp = Input(shape=(self.env_dim,))
        x = Dense(256, activation='relu')(inp)
        x = GaussianNoise(1.0)(x)
        #x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        out = Dense(self.act_dim, activation='tanh',
                    kernel_initializer=RandomUniform())(x)
        out = Lambda(lambda i: i*self.act_range)(out)
        model = Model(inp, out)
        model.summary()
        plot_model(model, 'actor_model.png', show_shapes=True)
        return model

    def target_predict(self, state):
        return self.target_model.predict(state)

    def transfer_weights(self):
        W = self.model.get_weights()
        target_W = self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * \
                                                        target_W[i]
        self.target_model.set_weights(target_W)

    # train actor to maximize Q-value
    def train(self, states, actions, grads):
        self.optimizer([states, grads])

    def optimizer(self, states, grads):
        # Actor Optimizer
        action_grads = K.placeholder(shape=(None, self.act_dim))

        # dA/d(theta) * dC/dA
        params_grads = tf.gradients(self.model.output,
                                    self.model.trainable_weights,
                                    -action_grads)
        grads = zip(params_grads, self.model.trainable_weights)
        return K.function([self.model.input, action_grads],
                          [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])

    def save(self, path):
        self.model.save_weights(path+'_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)

########################################################
## Critic
# Approximates Q-value function
###############
class Critic:
    def __init__(self, inp_dim, out_dim, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau = tau
        self.lr = lr

        self.model = self._build_net()
        self.target_model = self._build_net()

        self.action_grads = K.function([self.model.input[0],
                                        self.model.input[1]],
                                       K.gradients(self.model.output,
                                                  [self.model.input[1]]))
    def _build_net(self):
        state = Input(shape=(self.env_dim,))
        action = Input(shape=(self.act_dim,))
        x = Dense(256, activation='relu')(state)
        x = concatenate([x, action]) 
        x = Dense(128, activation='relu')(x)
        out = Dense(1, activation='linear',
                    kernel_initializer=RandomUniform())(x)
        model = Model([state, action], out)
        model.compile(Adam(self.lr), 'mse')
        model.summary()
        plot_model(model, 'critic_model.png', show_shapes=True)
        return model


    def gradients(self, states, actions):
        # Compute Q-value gradients wrt states and action policies
        return self.action_grads([states, actions])

    def target_predict(self, inp):
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        print('states shape:{}'.format(np.shape(states)))
        print('actions shape:{}'.format(np.shape(actions)))
        self.model.train_on_batch([states, actions], critic_target)

    def transfer_weights(self):
        W = self.model.get_weights()
        target_W = self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * \
                                                    target_W[i]
        self.target_model.set_weights(target_W)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)


###################################################
class DDPG:
    def __init__(self, state_size, action_size, action_bound, 
                buffer_size = 20000,
                gamma = 0.99,
                actor_lr = 0.00005,
                 critic_lr = 0.001,
                 tau = 0.001):
        self.action_size = action_size
        self.action_bound = action_bound
        self.state_size = state_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buffer_size = buffer_size

        # create replay memory buffer
        self.memory = deque(maxlen=self.buffer_size)

        # create actor & critic networks
        self.actor = Actor(state_size, action_size, action_bound,
                                                        self.actor_lr, tau)
        self.critic = Critic(state_size, action_size, self.critic_lr, tau)

    def policy_action(self, state):
        return self.actor.model.predict(state)

    def bellman_critic_target(self, rewards, q_values, dones):
        # use bellman equation to compute critic target
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * \
                                                        q_values[i]
        return critic_target

    def memorize(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def sample_batch(self, batch_size):
        
        minibatch = random.sample(self.memory, batch_size)

        state_batch = np.zeros((batch_size, self.state_size))
        next_state_batch = np.zeros((batch_size, self.state_size))
        action_batch = np.zeros((batch_size, self.action_size))
        reward_batch, done_batch = [], []

        for i in range(batch_size):
            state_batch[i] = minibatch[i][0]  
            action_batch[i] = minibatch[i][1]
            reward_batch.append(minibatch[i][2])
            next_state_batch[i] = minibatch[i][3]  
            done_batch.append(minibatch[i][4])

        return state_batch, action_batch, reward_batch, \
                                next_state_batch, done_batch


    def update_models(self, states, actions, critic_target):
        """
        Update actor and critic models from sampled experience
        """
        # train critic
        self.critic.train_on_batch(states, actions, critic_target)

        #Q-value gradients under current policy: dq/da
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)

        # train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1,
                                                                  self.action_size)))
        # transfer weights to target networks through polyak averaging
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self, env, summary_writer, gather_stats=True,
              render=False, max_episodes=3000, batch_size=64):
        results = []

        # First, gather experience
        tqdm_e = tqdm(range(max_episodes), desc='Score',
                      leave=True, unit=' episodes')
        for e in tqdm_e:
            time, cumul_reward, done = 0, 0, False
            state = env.reset().reshape([1, state_size])
            #actions, states, rewards = [], [], []
            noise = OrnsteinUhlenbeckProcess(size=self.action_size)

            while not done:

                if render: env.render()

                # Actor picks action following deterministic policy
                a = self.policy_action(state)
                #print('action_shape:{}'.format(np.shape(a)))
                # clip continuous values to be valid wrt environment
                a = np.clip(a+noise.generate(time),
                            -self.action_bound, self.action_bound)
                # obtain new states, rewards
                new_state, r, done, _ = env.step(a)

                new_state = np.reshape(new_state, [1, state_size])
                #print('shape of new_state:{}'.format(np.shape(new_state)))

                # add outputs to memory buffer
                self.memorize(state, a, r, new_state, done)

                #pdb.set_trace()
                # sample experience from memory buffer
                if len(self.memory) > batch_size:
                    states, actions, rewards, new_states, dones = \
                                    self.sample_batch(batch_size)
                    #print('shape of new_states:{}'.format(np.shape(new_states)))
                    #print('shape of actions:{}'.format(np.shape(actions)))
                    #print('shape of rewards:{}'.format(np.shape(rewards)))
                    #input('Press Enter to continue')

                    # predict target q-values using target networks
                    q_values = self.critic.target_predict([new_states,
                                                        self.actor.target_predict(new_states)])

                    # compute critic target
                    critic_target = self.bellman_critic_target(rewards,
                                                    q_values, dones)

                    # train actor & critic on sample batches
                    self.update_models(states, actions, critic_target)

                # update current state
                state = new_state
                cumul_reward += r
                time += 1

            #Gather stats every episode for plotting
            if(gather_stats):
                mean,stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # export results for tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()
            # display score
            tqdm_e.set_description("Score: "+ str(cumul_reward))
            tqdm_e.refresh()

        return results

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)



#####################################
if __name__ == '__main__':

    set_session(get_session())

    summary_writer = tf.summary.FileWriter('./logs/ddpg_pendu/')
    
    env = gym.make('Pendulum-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.high.shape[0]
    action_bound = env.action_space.high

    agent = DDPG(state_size, action_size, action_bound)


    gather_states = True
    stats = agent.train(env, summary_writer, gather_stats)

    if gather_stats:
        df = pd.DataFrame(np.array(stats))
        df._to_csv('ddpg_logs.csv', header=['Episode, Mean, Stdev'],
                   float_format='%10.5f')

    exp_dir = './ddpg/models/'
    if not os.path.exist(exp_dir):
        os.makedirs(exp_dir)

    export_path = exp_dir
    agent.save_weights(exp_dir)
    env.env.close()





    





