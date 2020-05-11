import sys
import gym
import matplotlib.pyplot as plt 
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # hyper-parameters
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005


        # Create models
        self.actor = self._build_actor()
        self.critic = self._build_critic()

    def _build_actor(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size,\
                        activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', \
                      optimizer=Adam(lr=self.actor_lr))
        return model

    def _build_critic(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size,\
                        activation='relu'))
        model.add(Dense(self.value_size, activation='relu'))
        model.summary()
        model.compile(loss='mse',optimizer=Adam(lr=self.critic_lr))
        return model

    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]
    
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward 
        else:
            advantages[0][action] = reward + self.discount_factor * \
                        next_value  - value
            target[0][0] = reward + self.discount_factor * next_value 

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

###############################
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(1)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    max_episodes = 10000

    #create an A2C agent
    agent = A2CAgent(state_size, action_size)

    file = open("cp_a2c-2.txt",'w')
    scores, episodes = [], []
    avgscores = []
    avg100scores = []
    last100scores = deque(maxlen=100)
    for e in range(max_episodes):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or \
                    score == env.spec.max_episode_steps-1 else -100
            
            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                score = score if score == env.spec.max_episode_steps \
                                                else score + 100
                scores.append(score)
                last100scores.append(score)
                episodes.append(e)
                avgscores.append(np.mean(scores))
                avg100scores.append(np.mean(last100scores))
                file.write('{}\t{}\t{}\t{}\n'.format(\
                        e, score, np.mean(scores), \
                        np.mean(last100scores)))
                print('Episodes: {}, scores: {}, avgscores: {:.2f},\
                      avg100scores: {:.2f}'.format(e, score, \
                      np.mean(scores), np.mean(last100scores)))
                break
            # end of while loop

        if np.mean(last100scores) > env.spec.max_episode_steps-10:
            print('Problem solved in {} episodes'.format(e))
            break
        # end of for-loop
    file.close()


    # plotting
    plt.plot(episodes, scores, label='Scores')
    plt.plot(episodes, avgscores, label='Avg Scores')
    plt.plot(episodes, avg100scores, label='Avg100Scores')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.grid()
    plt.legend(loc='upper left')
    plt.savefig('./cp_a2c-2.png')
    plt.show()


                



