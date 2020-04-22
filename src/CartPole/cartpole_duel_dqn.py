'''
Cartpole - Duel DQN
'''

import gym
import random
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.utils import plot_model
from keras import backend as K
from keras import optimizers
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
import pickle

# disable deprecated warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# Check the GPU availability

print('TensorFlow Version: {}'.format(tf.__version__))

if tf.__version__ < "2.0":
  get_available_gpus()
else:
  device_name = tf.test.gpu_device_name()
  print(device_name)
  if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
  print('Found GPU at: {}'.format(device_name))

#################################################

class DQNAgent:
    def __init__(self, state_size, action_size,
                 memory_size = 2000,
                 batch_size = 48,
                 discount_factor = 0.9,
                 learning_rate = 0.001,
                 train_start = 1000,
                 epsilon = 1.0, # change this if you are loading saved weights
                 epsilon_decay_rate = 0.99,
                 ddqn_flag = True,
                 polyak_avg = False,
                 pa_tau = 0.1, # weightage for polyak averaging
                 duel_flag = True, # enable dueling architecture
                 dueling_option = 'avg',
                 load_weights_path = None,
                 load_exp_path = None):

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory_size = memory_size

        # Create replay memory to store experience
        self.memory = deque(maxlen=self.memory_size)

        self.train_start = train_start
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.ddqn = ddqn_flag
        self.dueling_option = dueling_option
        self.polyak_avg = polyak_avg
        self.pa_tau = pa_tau

        # create main model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()


        if load_weights_path is not None:
            self.model.load_weights(load_weights_path)
            print('Weights loaded from File')

        if load_exp_path is not None:
            with open(load_exp_path, 'rb') as file:
                self.memory = pickle.load(file)
                print('Experience loaded from file')

        # initially both models share same weight
        self.target_model.set_weights(self.model.get_weights())
        # class constructor ends here

  
  
    def display_model_info(self):
        print("**------------------**")
        if self.ddqn:
            print('Double DQN with Dueling')
            if self.polyak_avg:
                print('Implements Polyak Averaging')
                print('PA Weighting factor, tau: ', self.pa_tau)
            else:
                print('DQN with Dueling')
                self.polyak_avg = False 
                print('Dueling Option: ', self.dueling_option)
                print('Batch Size: ', self.batch_size)
                print('Replay Memory size: ', self.memory_size)
                print('Learning Rate: ', self.learning_rate)
                print('Discount Factor: ', self.discount_factor)
                print('Epsilon Decay Rate: ', self.epsilon_decay_rate)
                print('Train Start: ', self.train_start)
                print("**---------------------**")

  
    def _build_model(self):
        # Advantage network
        network_input = Input(shape=(self.state_size,), name='network_input')
        A1 = Dense(24, activation='relu', name='A1')(network_input)
        A2 = Dense(24, activation='relu', name ='A2')(A1)
        A3 = Dense(self.action_size, activation='linear', name='A3')(A2)

        # Value network
        #V1 = Dense(24, activation='relu', name='V1')(network_input)
        #V2 = Dense(10, activation='relu', name='V2')(A2)
        V3 = Dense(1, activation='linear', name='V3')(A2)

        if duel_flag:
            if self.dueling_option == 'avg':
                network_output = Lambda(lambda x: x[0] - K.mean(x[0]) + x[1],\
                                        output_shape=(self.action_size,), name='network_output')([A3,V3])
            elif self.dueling_option == 'max':
                network_output = Lambda(lambda x: x[0] - K.max(x[0]) + x[1],\
                                        output_shape=(self.action_size,), name='network_output')([A3,V3])
            elif self.dueling_option == 'naive':
                network_output = Lambda(lambda x: x[0] + x[1],\
                                        output_shape=(self.action_size,), name='network_output')([A3,V3])
            else:
                raise Exception('Invalid Dueling Option')
        else: # normal model 
            network_output = A3

        model = Model(network_input, network_output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        plot_model(model, to_file='/content/gdrive/My Drive/Colab_Models/model.png',\
                show_shapes=True, show_layer_names=True)
        return model

  
    # def update_target_network(self):
    #   ''' Simply copy the weights from original model'''
    #   self.target_model.set_weights(self.model.get_weights())

    def update_target_network(self):
        ''' Implements Polyak Averaging for weight update 
        in target network
        '''
        if self.ddqn and self.polyak_avg:
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i] * self.pa_tau + target_weights[i] * (1-self.pa_tau)
                self.target_model.set_weights(target_weights)
        else:
            self.target_model.set_weights(self.model.get_weights())

    def update_epsilon(self):
        '''
        Reduce exploration rate over time
        '''
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay_rate
        else:
            self.epsilon = 0.01

    def add_experience(self, state, action, reward, next_state, done):
        '''  Add experiences to the replay memory '''
        self.memory.append([state, action, reward, next_state, done])

    def select_action(self, state):
        '''Implements epsilon-greedy policy '''
        if (random.random() < self.epsilon):
            action = np.random.randint(0, self.action_size)
        else: 
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])
            return action

    def get_maxQvalue_nextstate(self, next_state):
        # max Q value among the next state's action
        if self.ddqn:
            # DDQN
            # Current Q network selects the action
            # a'_max = argmax_a' Q(s',a')
            action = np.argmax(self.model.predict(next_state)[0])
            # target Q network evaluates the action
            # Q_max = Q_target(s', a'_max)
            max_q_value = self.target_model.predict(next_state)[0][action]
        else: 
            # DQN chooses the max Q value among next actions
            # Selection and evaluation of action is on the target Q network
            # Q_max = max_a' Q_target(s', a')
            max_q_value = np.amax(self.target_model.predict(next_state)[0])

        return max_q_value

    def train_model(self):
        ''' Training on Mini-Batch with Experience Replay  '''

        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        current_state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        qValues = np.zeros((self.batch_size, self.action_size))

        #action, reward, done = [], [], []
        action = np.zeros(self.batch_size, dtype=int)
        reward = np.zeros(self.batch_size)
        done = np.zeros(self.batch_size, dtype=bool)

        for i in range(self.batch_size):
            current_state[i] = mini_batch[i][0]   # current_state
            action[i] = mini_batch[i][1]
            reward[i] = mini_batch[i][2]
            next_state[i] = mini_batch[i][3]  # next_state
            done[i] = mini_batch[i][4]

        qValues[i] = self.model.predict(\
                                        current_state[i].reshape(1,self.state_size))[0]
        max_qvalue_ns = self.get_maxQvalue_nextstate(\
                                                    next_state[i].reshape(1,self.state_size))

        if done[i]:
            qValues[i][action[i]] = reward[i]
        else:
            qValues[i][action[i]] = reward[i] + \
                    self.discount_factor * max_qvalue_ns

        # train the model
        self.model.fit(current_state, qValues, 
                    batch_size = self.batch_size,
                    epochs=1, verbose=0)

        # update epsilon with each training step
        self.update_epsilon()

############

if __name__ == "__main__":

  env = gym.make('CartPole-v0')

  max_episodes = 1000 # people are running for 3000 episodes
  targetNetworkUpdateFreq = 10
  train_start = 1000
  model_save_freq = 300


  last100Scores=deque(maxlen=100)
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n

  # disable interactive display of plots
  plt.ioff()

  # Create a DQN Agent
  deepQ = DQNAgent(state_size, action_size)

  deepQ.display_model_info()

  max_steps = 1
  stepCounter = 0
  Scores = []
  AvgScores = []
  Avg100Scores = []

  for e in range(max_episodes):

    state = env.reset().reshape(1,state_size)

    t = 0
    done = False
    while not done:
        #env.render()

      action = deepQ.select_action(state)
      next_state, reward, done, info = env.step(action)
      next_state = np.reshape(next_state, [1, state_size])

      # this is crucial for Cart Pole
      reward = reward if not done else -100

      # number of steps taken is our score
      t += 1

      # add experiences to replay memory
      deepQ.add_experience(state, action, reward, \
                           next_state, done)

      # experience replay - training
      deepQ.train_model()

        # if stepCounter % targetNetworkUpdateFreq == 0:
            #   deepQ.update_target_network()

      stepCounter += 1
      state = next_state

      if done:

        # update target network once in each episode
        deepQ.update_target_network()

        Scores.append(t)
        AvgScores.append(np.mean(Scores))
        last100Scores.append(t)
        Avg100Scores.append(np.mean(last100Scores))

        print('Episode: {}, time steps: {}, AvgScore: {:0.2f}, \
              epsilon: {:0.2f}, replay size: {}, Step Count: {}'.format(e, t, \
                                                                        np.mean(Scores), 
                                                                        deepQ.epsilon,
                                                                        len(deepQ.memory),
                                                                        stepCounter))
        # store into file
        with open('/content/gdrive/My Drive/Colab_Models/cp_result.txt','a+') as file2:
            file2.write('{}\t {} \t {:0.2f} \t {:0.2f}\t {}\n'\
                        .format(e, t, np.mean(Scores),\
                                np.mean(last100Scores), stepCounter))
            #print('wrote into file')
            break

    # time-step for-loop ends here
    # save best models 
    if t > max_steps:
        max_steps = t
        deepQ.model.save_weights('/content/gdrive/My Drive/Colab_Models/cp_model_steps_{}.h5'.format(t))



    # save model weights
    if e % model_save_freq == 0:
        deepQ.model.save_weights('cp_model_{}.h5'.format(e))


        with open("./saved_models/cp_exp_{}.txt".format(e), 'wb') as file:
            pickle.dump(deepQ.memory, file)

    if np.mean(last100Scores) > (env.spec.max_episode_steps-5):
        print('The problem is solved in {} episodes. Exiting'.format(e))
        break
  # episode loop ends here

  # plot
  plt.plot(Scores)
  plt.plot(AvgScores)
  plt.plot(Avg100Scores, 'm-')
  plt.xlabel('Episodes')
  plt.ylabel('Scores')
  plt.legend(['Actual', 'Average', 'Avg100Scores'])
  plt.savefig('cp_duel_dqn_plot.png')
  plt.show()

