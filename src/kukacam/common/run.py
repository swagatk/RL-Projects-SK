"""
Will be required if we want to keep the run function outside the RL agent class. 
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

# Local imports
from utils import uniquify

# validate function
def validate(env, agent, max_eps=50):
    ep_reward_list = []
    for ep in range(max_eps):
        state = env.reset()
        t = 0
        ep_reward = 0
        while True:
            action, _ = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            t += 1
            if done:
                ep_reward_list.append(ep_reward)
                break
    # outside for loop
    mean_ep_reward = np.mean(ep_reward_list)
    return mean_ep_reward

def datagen(env, agent, max_eps=50, save_path=None):

    feature_list = []
    action_list = []
    reward_list = []
    for _ in range(max_eps):
        obs = env.reset()
        state = np.asarray(obs, dtype=np.float32) / 255.0
        t = 0
        ep_reward = 0
        while True:
            f = agent.extract_feature(state)
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32) / 255.0
            
            feature_list.append(f)
            action_list.append(action)
            reward_list.append(reward)

            if save_path is not None:
                img_name = save_path + 'img_{}.png'.format(t)
                plt.savefig(img_name, format='png')

            state = next_state
            t += 1

            if done:
                break

    return np.array(feature_list), np.array(action_list), reward_list

def run_ipg(env, agent):

    if agent.filename is not None:
        agent.filename = uniquify(agent.path + agent.filename)

    # initial state and goal
    state = np.asarray(env.reset(), dtype=np.float32) / 255.0  # convert into float array
    goal = np.asarray(env.reset(), dtype=np.float32) / 255.0

    # store successful states
    if agent.her_strategy == 'success':
        desired_goals = deque(maxlen=1000)

    start = datetime.datetime.now()
    val_scores = []       # validation scores
    best_score = -np.inf
    s_scores = []               # all season scores
    ep_lens = []                # episode lengths 
    ep_scores = []              # episodic rewards
    self.episodes = 0           # global episode count
    for s in range(self.SEASONS):
        # discard trajectories from previous season
        states, next_states, actions, rewards, dones, goals = [], [], [], [], [], []
        ep_experience = []     # episodic experience buffer
        s_score = 0         # season score
        ep_score = 0        # episodic reward
        ep_cnt = 0          # episodes in each season
        ep_len = 0          # length of each episode
        done = False
        for _ in range(self.training_batch):    # time steps
            action = self.policy(state, goal)
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32) / 255.0

            if self.her_strategy == 'success': 
                if reward == 1:
                    desired_goals.append([state, action, reward, next_state, done, goal])

            # this is used for on-policy training
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            goals.append(goal)

            # store in replay buffer for off-policy training
            self.buffer.record([state, action, reward, next_state, done, goal])
            # Also store in a separate buffer
            ep_experience.append([state, action, reward, next_state, done, goal])

            state = next_state
            ep_score += reward
            ep_len += 1

            if done:

                # HER: Final state strategy
                if self.her_strategy == 'final':
                    hind_goal = ep_experience[-1][3]        # final state strategy
                elif self.her_strategy == 'success': 
                    if len(desired_goals) < 1:
                        hind_goal = ep_experience[-1][3]  
                    else:   # random successful states as a goal
                        index = np.random.choice(len(desired_goals))
                        hind_goal = desired_goals[index][3]
                elif self.her_strategy == 'future':   # future state strategy
                    hind_goal = None
                else:
                    raise ValueError("Invalid choice for her strategy. Exiting ...")

                # Add hindsight experience to the buffer
                # in this case, use last state as the goal_state
                # here ep_experience buffer is cleared at the end of each episode.
                self.add_her_experience(ep_experience, hind_goal, extract_feature=False)
                # clear the local experience buffer
                ep_experience = []

                self.episodes += 1
                s_score += ep_score        # season score
                ep_cnt += 1             # episode count in a season
                ep_scores.append(ep_score)
                ep_lens.append(ep_len)

                if self.WB_LOG:
                    wandb.log({'time_steps' : self.time_steps,
                        'Episodes' : self.episodes, 
                        'mean_ep_score': np.mean(ep_scores),
                        'mean_ep_len' : np.mean(ep_lens)})

                # prepare for next episode
                state = self.env.reset()
                state = np.asarray(state, dtype=np.float32) / 255.0
                goal = self.env.reset()
                goal = np.asarray(goal, dtype=np.float32) / 255.0
                ep_len, ep_score = 0, 0
                done = False
            # end of done block
        # end of for training_batch loop

        # Add hindsight experience to the buffer
        # here we are using random goal states
        # self.add_her_experience(ep_experience)
        # clear the local experience buffer
        # ep_experience = []          # not required as we are clearing it for every season.

        # on-policy & off-policy training
        actor_loss, critic_loss = self.train(states, actions, rewards, next_states, dones, goals)

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
            os.makedirs(best_model_path, exist_ok=True)
            self.save_model(best_model_path)
            print('Season: {}, Update best score: {}-->{}, Model saved!'.format(s, best_score, mean_s_score))
            best_score = mean_s_score
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
                        'ep_per_season' : ep_cnt, 
                        'Season' : s})

        if self.chkpt_freq is not None and s % self.chkpt_freq == 0:          
            chkpt_path = self.path + 'chkpt_{}/'.format(s)
            os.makedirs(chkpt_path, exist_ok=True)
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

    # end of season loop
    end = datetime.datetime.now()
    print('Time to completion: {}'.format(end-start))
    print('Mean episodic score over {} episodes: {:.2f}'.format(self.episodes, np.mean(ep_scores)))
    self.env.close()
# Main training function
# def run(env, agent):

#     if load_path is not None:
#         agent.load_model(load_path)
#         print('---Model parameters are loaded---')

#     # create folder for storing result files
#     if LOG_FILE:
#         os.makedirs(save_path, exist_ok=True)
#         tag = '_her' if use_HER else ''
#         filename = env_name + '_' + config_dict['algo'] + tag + '.txt'
#         filename = uniquify(save_path + filename)

#     val_scores = []
#     val_score = 0

#     # initial state
#     obs = env.reset()
#     state = np.asarray(obs, dtype=np.float32) / 255.0
#     if use_HER:
#         goal = np.asarray(env.reset(), dtype=np.float32) / 255.0

#     start = datetime.datetime.now()
#     best_score = -np.inf
#     ep_lens = []  # episodic length
#     ep_scores = []      # All episodic scores
#     s_scores = []       # season scores
#     total_ep_cnt = 0  # total episode count
#     global_time_steps = 0   # global training step counts
#     for s in range(seasons):
#         states, next_states, actions, rewards, dones = [], [], [], [], []

#         if use_HER:
#             goals = []
#             temp_experience = []      # temporary experience buffer

#         s_score = 0     # season score
#         ep_cnt = 0      # no. of episodes in each season
#         ep_steps = 0    # no. of steps in each episode
#         ep_score = 0    # episodic reward
#         done = False
#         for t in range(config_dict['training_batch']):
#             if use_HER:
#                 action, _ = agent.policy(state, goal)
#             else:
#                 action, _ = agent.policy(state)

#             next_obs, reward, done, _ = env.step(action)
#             next_state = np.asarray(next_obs, dtype=np.float32) / 255.0

#             # this is used for on-policy training
#             states.append(state)
#             next_states.append(next_state)
#             actions.append(action)
#             rewards.append(reward)
#             dones.append(done)

#             # store in replay buffer for off-policy training
#             if use_HER:
#                 goals.append(goal)
#                 agent.buffer.record([state, action, reward, next_state, done, goal]) # check this
#                 # Also store in a separate temporary buffer
#                 temp_experience.append([state, action, reward, next_state, done, goal])
#             else:
#                 agent.buffer.record([state, action, reward, next_state, done])

#             state = next_state
#             ep_score += reward
#             ep_steps += 1       
#             global_time_steps += 1

#             if done:
#                 s_score += ep_score
#                 ep_cnt += 1         # episode count in each season
#                 total_ep_cnt += 1   # global episode count
#                 ep_scores.append(ep_score)
#                 ep_lens.append(ep_steps)

#                 if use_HER:
#                     hind_goal = temp_experience[-1][3]  # Final state strategy
#                     # add hindsight experience to the main buffer
#                     agent.add_her_experience(temp_experience, hind_goal)
#                     temp_experience = []    # clear the temporary buffer

#                 # off-policy training after each episode
#                 if config_dict['algo'] == 'sac_her':
#                     a_loss, c_loss, alpha_loss = agent.train()
#                     if WB_LOG:
#                         wandb.log({'time_steps' : global_time_steps,
#                             'Episodes' : total_ep_cnt,
#                             'mean_ep_score': np.mean(ep_scores),
#                             'ep_actor_loss' : a_loss,
#                             'ep_critic_loss' : c_loss,
#                             'ep_alpha_loss' : alpha_loss,
#                             'mean_ep_len' : np.mean(ep_lens)},
#                             step = total_ep_cnt)

#                 # prepare for next episode
#                 state = np.asarray(env.reset(), dtype=np.float32) / 255.0
#                 if use_HER: 
#                     goal = np.asarray(env.reset(), dtype=np.float32) / 255.0
#                 ep_steps, ep_score = 0, 0
#                 done = False

#             # done block ends here
#         # end of one season

#         s_score = np.mean(ep_scores[-ep_cnt:])  # mean of last ep_cnt episodes
#         s_scores.append(s_score)
#         mean_s_score = np.mean(s_scores)
#         mean_ep_score = np.mean(ep_scores)
#         mean_ep_len = np.mean(ep_lens)

#         if  mean_s_score > best_score:
#             agent.save_model(save_path)
#             print('Season: {}, Update best score: {}-->{}, Model saved!'.format(s, best_score, mean_s_score))
#             best_score = mean_s_score

#         if val_freq is not None:
#             if total_ep_cnt % val_freq == 0:
#                 print('Episode: {}, Score: {}, Mean score: {}'.format(total_ep_cnt, ep_score, mean_ep_score))
#                 val_score = validate(env)
#                 val_scores.append(val_score)
#                 mean_val_score = np.mean(val_scores)
#                 print('Episode: {}, Validation Score: {}, Mean Validation Score: {}' \
#                       .format(total_ep_cnt, val_score, mean_val_score))
#                 if WB_LOG:
#                     wandb.log({'val_score': val_score, 
#                                 'mean_val_score': val_score})

#         if WB_LOG:
#             wandb.log({'Season Score' : s_score, 
#                         'Mean Season Score' : mean_s_score,
#                         'Actor Loss' : a_loss,
#                         'Critic Loss' : c_loss,
#                         'Mean episode length' : mean_ep_len,
#                         'Season' : s})

#         if LOG_FILE:
#             if config_dict['algo'] == 'sac_her':
#                 with open(filename, 'a') as file:
#                     file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
#                             .format(s, total_ep_cnt, global_time_steps, mean_ep_len,
#                                     s_score, mean_s_score, a_loss, c_loss, alpha_loss))
#             elif config_dict['algo'] == 'ipg':
#                 with open(filename, 'a') as file:
#                     file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
#                             .format(s, total_ep_cnt, global_time_steps, mean_ep_len,
#                                     s_score, mean_s_score, a_loss, c_loss))

#         if success_value is not None:
#             if best_score > success_value:
#                 print('Problem is solved in {} seasons with best score {}'.format(s, best_score))
#                 print('Mean season score: {}'.format(mean_s_score))
#                 break
#     # end of episode-loop
#     end = datetime.datetime.now()
#     print('Time to Completion: {}'.format(end - start))
#     env.close()
#     print('Mean episodic score over {} episodes: {:.2f}'.format(total_ep_cnt, np.mean(ep_scores)))

#     if COLAB:
#         p.disconnect(p.DIRECT) # on google colab
#     # end of run function