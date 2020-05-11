import gym
import numpy as np
from collections import deque
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Lambda, Add
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import matplotlib.pyplot as plt
 

class REINFORCEAgent:
    def __init__(self, state_size, action_size):

        self.learning_rate = 0.0001
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99
        

        self.model = self._build_model()

        self.states = []
        self.actions = []
        self.rewards = []

    # approximate policy using a Neural Network
    # state is input and probability of each action is output
    def _build_model(self):
        model= Sequential()
        model.add(Dense(24, input_dim=self.state_size,
                        activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p = policy)[0]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + \
                                                    rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards 

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


    def train_model(self):
        episode_length = len(self.states)
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))
        
        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    scores, episodes = [], []
    avgscores = []
    avg100scores = []
    last100scores = deque(maxlen=100)
    max_episodes = 10000

    agent = REINFORCEAgent(state_size, action_size)

    for e in range (max_episodes):
        done = False
        score = 0
        state = env.reset().reshape(1, state_size)

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or \
                    score == env.spec.max_episode_steps-1 else -100

            agent.append_sample(state, action, reward)

            score += reward
            state = next_state
            if done:
                agent.train_model()
                score = score if score == env.spec.max_episode_steps \
                                else score + 100
                scores.append(score)
                episodes.append(e)
                avgscores.append(np.mean(scores))
                last100scores.append(score)
                avg100scores.append(np.mean(last100scores))
                break
            # end of while loop
        if e % 100 == 0:
            print('episode: {}, score: {}, avg score: {:.2f},\
                    avg100score: {:.2f}'\
                    .format(e, score, np.mean(scores),
                            np.mean(last100scores)))
        if np.mean(last100scores) > (env.spec.max_episode_steps-5):
            print('problem solved in {} steps. exiting'.format(e))
            break
        # end of for loop

    # Plotting
    plt.plot(episodes, scores, label='scores')
    plt.plot(episodes, avgscores, label='avg scores')
    plt.plot(episodes, avg100scores, label = 'avg100scores')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid()
    plt.savefig('./cp_pg.png')
    plt.show()





