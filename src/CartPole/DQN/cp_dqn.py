'''
Solving Cartpole problem with DQN
- It uses new implementation available within the algo folder
- Successfully tested on 24/03/204
- Uses new Buffer
'''
import numpy as np
import gymnasium as gym
import sys 

sys.path.append('/home/kumars/RL-Projects-SK')
from src.algo.dqn import DQNAgent, DQNPERAgent 

##############################################
def train(env, agent, max_episodes=300, 
          train_freq=1, copy_freq=1):
    
    file = open('cp_dqn_per.txt', 'w')
    
    # polyak averaging factor
    tau = 0.1 if copy_freq < 10 else 1.0

    best_score = 0
    scores = []
    avg_scores, avg100_scores = [], []
    global_step_cnt = 0
    for e in range(max_episodes):
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        done = False
        ep_reward = 0
        t = 0
        while not done:
            global_step_cnt += 1
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -100
            next_state = np.expand_dims(next_state, axis=0) # (-1, 4)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            t += 1

            # train
            if global_step_cnt % train_freq == 0:
                agent.experience_replay()

            # update target model
            if global_step_cnt % copy_freq == 0:
                agent.update_target_model(tau=tau)
                
            # episode ends here
        if e > 100 and t > best_score:
            agent.save_model('best_model_per.h5')
            best_score = t
        scores.append(t)
        avg_scores.append(np.mean(scores))
        avg100_scores.append(np.mean(scores[-100:]))
        file.write(f'{e}\t{t}\t{np.mean(scores)}\t{np.mean(scores[-100:])}\n')
        if e % 20 == 0:
            print(f'e:{e}, episodic reward: {t}, avg ep reward: {np.mean(scores):.2f}')
    print('End of training')
    file.close()
############################################
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # create a dqn agent
    # agent = DQNAgent(obs_shape, n_actions,
    #              buffer_size=2000,
    #              batch_size=24)

    agent = DQNPERAgent(obs_shape, n_actions,
                 buffer_size=20000,
                 batch_size=64)

train(env, agent, max_episodes=200, copy_freq=100)

