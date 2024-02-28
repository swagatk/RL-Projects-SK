import sys
import os
import random
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from common.buffer import ReplayBuffer

sys.path.append('../common/')
from sumtree import STBuffer

class DQNAgent:
  def __init__(self, obs_shape: tuple, n_actions: int,
               buffer_size=2000, batch_size=24,
               grad_tape=False,
               model=None):

    self.obs_shape = obs_shape  # shape: tuple
    self.action_size = n_actions  # number of discrete action state (int)



    # hyper parameters for DQN
    self.gamma =  0.9         # discount factor
    self.epsilon = 1.0        # explore rate
    self.epsilon_decay = 0.99
    self.epsilon_min = 0.01
    self.batch_size = batch_size
    self.buffer_size = buffer_size  # replay buffer size
    self.grad_tape = grad_tape
    self.train_start = 1000     # minimum buffer size to start training
    self.learning_rate = 0.001


    # create replay memory using deque
    self.memory = ReplayBuffer(self.obs_shape,
                               (1, ),
                               self.buffer_size)

    # create main model and target model
    if model is None:
      if self.grad_tape: # use gradient tape
        self.model = self._build_model_2()
        self.target_model = self._build_model_2()
        self.optimizer = tf.keras.optimizers.Adam()
      else:
        self.model = self._build_model()
        self.target_model = self._build_model()
    else:
      self.model = model
      self.target_model = tf.keras.models.clone_model(model)
    self.model.summary()

    # initialize target model
    self.target_model.set_weights(self.model.get_weights())

  # approximate Q-function with a Neural Network
  def _build_model(self):
    model = keras.Sequential([
        keras.layers.Dense(24, input_shape=self.obs_shape, activation='relu',
                    kernel_initializer='he_uniform'),
        keras.layers.Dense(24, activation='relu',
                           kernel_initializer='he_uniform'),
        keras.layers.Dense(self.action_size, activation='linear',
                    kernel_initializer='he_uniform')
    ])
    model.summary()
    model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    return model

  def _build_model_2(self):
    input = keras.layers.Input(shape=self.obs_shape)
    x = keras.layers.Dense(24, activation='relu',
                           kernel_initializer='he_uniform')(input)
    x = keras.layers.Dense(24, activation='relu',
                           kernel_initializer='he_uniform')(x)
    y = keras.layers.Dense(self.action_size, activation='linear',
                           kernel_initializer='he_uniform')(x)
    model = keras.Model(inputs=input, outputs=y)
    return model

  def update_target_model(self, tau=0.01):
    model_weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    # Ensure shapes match
    if len(model_weights) != len(target_weights):
      raise ValueError('Model and Target should have same number of items')

    # update weights layer-by-layer using Polyak Averaging
    new_weights = []
    for w, w_dash in zip(model_weights, target_weights):
      new_w = tau * w + (1 - tau) * w_dash
      new_weights.append(new_w)
    self.target_model.set_weights(new_weights)


  # get action from the main model using epsilon-greedy policy
  def get_action(self, state):
    if np.random.rand() <= self.epsilon: # explore
      return random.randrange(self.action_size)
    else:
      q_value = self.model.predict(state, verbose=0)  # exploit
      return np.argmax(q_value[0])

  # save sample <s, a, r, s'>. into replay memory
  def store_experience(self, state, action, reward, next_state, done):
    self.memory.add(state, action, reward, next_state, done)


  # implements DDQN algorithm
  def get_target_q_value(self, next_state):
    # Current Q network selects action for next_state
    action = np.argmax(self.model.predict(next_state, verbose=0)[0])
    # target Q network evaluates the action
    # Q_max = Q_target(s', a'_max)
    max_q_value = self.target_model.predict(next_state, verbose=0)[0][action]
    return max_q_value

  def get_target_q_value_2(self, next_states): # batch input

    q_values_ns = self.model.predict(next_states, verbose=0)

    # main model is used for action selection
    max_actions = np.argmax(q_values_ns, axis=1)

    # target model is used for action evaluation
    target_q_values_ns = self.target_model.predict(next_states, verbose=0)

    max_q_values = target_q_values_ns[range(len(target_q_values_ns)), max_actions]

    return max_q_values



  def experience_replay_2(self):  # use gradient tape
    if len(self.memory) < self.train_start:
      return

    batch_size = min(self.batch_size, len(self.memory))
    mini_batch = self.memory.sample(batch_size) # replay buffer

    y_pred, y_target = [], []
    #for state, action, reward, next_state, done in mini_batch:
    for state, action, reward, next_state, done in zip(*mini_batch):

      # q-value prediction for a given state
      q_values_cs = self.model (state, verbose=0)
      y_pred.append(q_values_cs[0])  # model prediction

      # target q-value
      max_q_value_ns = self.get_target_q_value(next_state)

      action = action.astype(int)[0]
      done = done.astype(bool)[0]
      reward = reward[0]
      # correction on the Q value for the action used
      if done:
        q_values_cs[0][action] = reward
      else:
        q_values_cs[0][action] = reward + \
                          self.gamma * max_q_value_ns
      y_target.append(q_values_cs[0])

    # outside for loop
    #ipdb.set_trace()
    pred = tf.stack(y_pred)
    target = tf.stack(y_target)
    with tf.GradientTape() as tape:
      loss = tf.reduce_mean((pred - target)**2)

    w = self.model.trainable_variables
    # compute gradients
    grads = tape.gradient(loss, w)
    self.optimizer.apply_gradients(zip(grads, w))

    # decay epsilon
    self.update_epsilon()

  def experience_replay_3(self):  # uses new replay buffer
    if len(self.memory) < self.train_start:
      return

    batch_size = min(self.batch_size, len(self.memory))
    mini_batch = self.memory.sample(batch_size) # uses replay buffer

    state_batch, q_value_batch = [], []
    for state, action, reward, next_state, done in zip(*mini_batch):
      state = np.expand_dims(state, axis=0)
      next_state = np.expand_dims(next_state, axis=0)
      q_values_cs = self.model.predict(state, verbose=0)
      max_q_value_ns = self.get_target_q_value(next_state)

      action = action.astype(int)[0] # check
      done = done.astype(bool)[0] # check
      reward = reward[0] # check
      if done:
        q_values_cs[0][action] = reward
      else:
        q_values_cs[0][action] = reward + self.gamma * max_q_value_ns

      state_batch.append(state[0])
      q_value_batch.append(q_values_cs[0])

    # train the Q network
    self.model.fit(np.array(state_batch),
                   np.array(q_value_batch),
                   batch_size = batch_size,
                   epochs = 1,
                   verbose = 0)

    # decay epsilon over time
    self.update_epsilon()

  def experience_replay(self):
    # uses new replay buffer
    # computationally efficient
    # v3 & v4 are identical
    if len(self.memory) < self.train_start:
      return

    #ipdb.set_trace()
    batch_size = min(self.batch_size, len(self.memory))
    mini_batch = self.memory.sample(batch_size) # uses replay buffer

    states, actions, rewards, next_states, dones = mini_batch
    q_values_cs = self.model.predict(states, verbose=0)
    max_q_values_ns = self.get_target_q_value_2(next_states)

    for i in range(len(q_values_cs)):
      action = actions[i].astype(int)[0] # check
      done = dones[i].astype(bool)[0] # check
      reward = rewards[i][0] # check
      if done:
        q_values_cs[i][action] = reward
      else:
        q_values_cs[i][action] = reward + self.gamma * max_q_values_ns[i]

    # train the Q network
    self.model.fit(np.array(states),
                   np.array(q_values_cs),
                   batch_size = batch_size,
                   epochs = 1,
                   verbose = 0)

    # decay epsilon over time
    self.update_epsilon()


  # decrease exploration, increase exploitation
  def update_epsilon(self):
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def save_model(self, filename: str):
    self.model.save_weights(filename)

  def load_model(self, filename: str):
    self.model.load_weights(filename)


class DQNPERAgent(DQNAgent):

    def __init__(self, obs_shape: tuple, n_actions: int,
                        buffer_size=2000, batch_size=24,
                        grad_tape=False, model=None):
        super().__init__(obs_shape, n_actions, buffer_size, 
                        batch_size, grad_tape, model)

        # uses a sumtree Buffer
        self.memory = STBuffer(capacity=buffer_size)


    def store_experience(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def experience_replay(self):
        tree_idx, mini_batch = self.memory.sample(self.batch_size)

        states = mini_batch[:, 0] 
        actions = mini_batch[:,1]
        rewards = mini_batch[:, 2]
        next_states = mini_batch[:, 3]
        dones  = mini_batch[:, 4]

        q_values_cs = self.model.predict(states)
        q_values_cs_old = np.array(q_value_cs).copy() # deep copy 
        max_q_value_ns = self.get_target_q_value_2(next_states)


        for i in range(len(q_values_cs)):
            action = actions[i].astype(int)[0] # check
            done = dones[i].astype(bool)[0] # check
            reward = rewards[i][0] # check
            if done:
                q_values_cs[i][action] = reward
            else:
                q_values_cs[i][action] = reward + self.gamma * max_q_values_ns[i]

        # update experience priorities
        indices = np.arange(self.batch_size, dtype=np.int32)
        absolute_errors np.abs(q_values_cs_old[indices, actions] - \
                                q_values_cs[indices, actions])
        self.memory.batch_update(tree_idx, absolute_errors)

        # train the Q network
        self.model.fit(np.array(states),
                    np.array(q_values_cs),
                    batch_size = batch_size,
                    epochs = 1,
                    verbose = 0)

        # decay epsilon over time
        self.update_epsilon()
