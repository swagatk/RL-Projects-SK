import os
import random
import gym
import pylab
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import matplotlib.pyplot as plt
from sumtree import SumTree, Memory

class DQNAgent:
    def __init__(self, state_size, action_size,
                 memory_size = 2000,
                 batch_size = 24,
                 discount_factor = 0.95,
                 learning_rate = 0.001, #0.00025,
                 train_start = 1000,
                 epsilon_start = 1.0,  # exploration rate
                 epsilon_min = 0.01,
                 epsilon_decay_rate = 0.99, #0.9995,
                 ddqn_flag = True, # double DQN
                 duel_flag = False,  # use dueling architecture
                 use_PER = True,     # use Prioritized experience replay
                 polyak_avg = False,  # use Soft update
                 pa_tau = 0.1, # weightage for polyak averaging
                 dueling_option = 'avg',
                 epsilon_greedy_strategy = False, 
                 load_weights_path = None,
                 load_exp_path = None):
    
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.train_start = train_start
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min 
        self.epsilon_decay_rate = epsilon_decay_rate
        self.ddqn = ddqn_flag
        self.dueling_option = dueling_option
        self.polyak_avg = polyak_avg
        self.pa_tau = pa_tau
        self.use_PER = use_PER
        self.dueling = duel_flag  
        self.epsilon_gs = epsilon_greedy_strategy 

        # Create replay memory to store experience
        if self.use_PER:
            self.memory = Memory(self.memory_size)
        else:
            self.memory = deque(maxlen=self.memory_size)

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
        print('Double DQN:', self.ddqn)
        print('Dueling:', self.dueling)
        print('Soft Update:', self.polyak_avg)
        print('PER:', self.use_PER)

        if self.polyak_avg:
            print('Tau: ', self.pa_tau)
            if self.dueling: 
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

        if self.dueling:
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
        # plot_model(model, to_file='/content/gdrive/My Drive/Colab_Models/model.png',\
        #            show_shapes=True, show_layer_names=True)
        return model

    def _build_model2(self):

        # Advantage network
        network_input = Input(shape=(self.state_size,), name='network_input')
        X = Dense(512, activation='relu', kernel_initializer='he_uniform')(network_input)
        X = Dense(256, activation='relu', kernel_initializer='he_uniform')(X)
        X = Dense(64, activation='relu', kernel_initializer='he_uniform')(X)
        A = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(X)

        # Value network
        #V1 = Dense(24, activation='relu', name='V1')(network_input)
        #V2 = Dense(10, activation='relu', name='V2')(A2)
        V = Dense(1, activation='linear', kernel_initializer='he_uniform')(X)

        if self.dueling:
            if self.dueling_option == 'avg':
                network_output = Lambda(lambda x: x[0] - K.mean(x[0]) + x[1],\
                        output_shape=(self.action_size,), name='network_output')([A,V])
            elif self.dueling_option == 'max':
                network_output = Lambda(lambda x: x[0] - K.max(x[0]) + x[1],\
                        output_shape=(self.action_size,), name='network_output')([A,V])
            elif self.dueling_option == 'naive':
                network_output = Lambda(lambda x: x[0] + x[1],\
                        output_shape=(self.action_size,), name='network_output')([A,V])
            else:
                raise Exception('Invalid Dueling Option')
        else: # normal model 
            network_output = A

        model = Model(network_input, network_output)
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.compile(loss='mse', \
                      optimizer=RMSprop(lr=self.learning_rate,
                                        rho=0.95, epsilon=0.01),\
                      metrics=["accuracy"])
        model.summary()                                 
        # plot_model(model, to_file='/content/gdrive/My Drive/Colab_Models/model.png',\
        #            show_shapes=True, show_layer_names=True)
        return model
    
    def update_target_network(self):
        ''' Implements Polyak Averaging for weight update 
        in target network - Soft target model update
        '''
        if self.ddqn and self.polyak_avg:
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i] * self.pa_tau + \
                target_weights[i] * (1-self.pa_tau)
                # for-loop ends here
            self.target_model.set_weights(target_weights)
        else:
            self.target_model.set_weights(self.model.get_weights())

    def update_epsilon(self):
        '''
        Reduce exploration rate over time
        '''
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_rate
        else:
            self.epsilon = self.epsilon_min


    def add_experience(self, state, action, reward, next_state, done):
        '''  Add experiences to the replay memory '''
        experience = [state, action, reward, next_state, done]
        if self.use_PER:
            self.memory.store(experience)
        else:
            #self.memory.append((state, action, reward, next_state, done))
            self.memory.append(experience)

    def select_action(self, state):
        '''Implements epsilon-greedy policy '''
        if (random.random() < self.epsilon): # explore
            action = np.random.randint(0, self.action_size)
        else: # exploit 
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])
            #print('shape of q_values:{}'.format(np.shape(q_values)))
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
        if self.use_PER:
            tree_idx, mini_batch = self.memory.sample(self.batch_size)
        else:
            if len(self.memory) < self.train_start:
                return
            else:
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

            qValues[i] = self.model.predict(current_state[i].reshape(1,self.state_size))[0]
            max_qvalue_ns = self.get_maxQvalue_nextstate(next_state[i].reshape(1,self.state_size))

            if done[i]:
                qValues[i][action[i]] = reward[i]
            else:
                qValues[i][action[i]] = reward[i] + \
                        self.discount_factor * max_qvalue_ns

        if self.use_PER:
            target_old = np.array(self.model.predict(current_state))
            target = qValues
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, \
                    np.array(action)]- target[indices, np.array(action)])
            # Update priority
            self.memory.batch_update(tree_idx, absolute_errors)

        # train the model
        self.model.fit(current_state, qValues, 
                       batch_size = self.batch_size,
                       epochs=1, verbose=0)
        # update epsilon with each training step
        self.update_epsilon()

###########
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env.seed(0)  # use a fixed seed 
    max_episodes = 1000
    train_start = 1000

    last100Scores = deque(maxlen=100)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

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

            # Act & get reward
            action = deepQ.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            if not done or t == env._max_episode_steps - 1:
                reward = reward
            else:
                reward = -100

            t += 1

            # Remember
            deepQ.add_experience(state,action,reward, next_state, done)

            # Experience replay
            deepQ.train_model()

            stepCounter += 1
            state = next_state

            if done:
                deepQ.update_target_network()

                Scores.append(t)
                AvgScores.append(np.mean(Scores))
                last100Scores.append(t)
                Avg100Scores.append(np.mean(last100Scores))

                print('Episode: {}, time steps: {}, AvgScore: {:0.2f},\
                    Avg100Score: {:0.2f}, epsilon: {:0.2f}'\
                    .format(e, t, np.mean(Scores), np.mean(last100Scores),\
                            deepQ.epsilon)  )

                with open('./data2/cp_ddqn_PER_bs24:24:24-3.txt','a+') as file2:
                    file2.write('{}\t {} \t {:0.2f} \t {:0.2f}\t {}\n'\
                                .format(e, t, np.mean(Scores),\
                                        np.mean(last100Scores), stepCounter))
                    #print('wrote into file')
                    break
                # while loop ends here

        #if np.mean(last100Scores) > (env.spec.max_episode_steps-5):
        #    print('The problem is solved in {} episodes. Exiting'.format(e))
        #    break
        # for-episode-loop ends here

    # Plot
    plt.plot(Scores)
    plt.plot(AvgScores)
    plt.plot(Avg100Scores, 'm-')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.legend(['Actual', 'Average', 'Avg100Scores'])
    plt.savefig('./img2/cp_ddqn_PER_bs24:24:24-3.png.png')
    plt.show()



