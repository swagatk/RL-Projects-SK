import sys
import os
import tensorflow as tf
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from ipg_her import IPGHERAgent
from IPG.ipg import IPGAgent



import numpy as np
import matplotlib.pyplot as plt
from common.grad_cam import save_and_display_gradcam, make_gradcam_heatmap


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print('GPU Name:', gpu.name, 'Device Type:', gpu.device_type)
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':

    config_dict = dict(
        lr_a = 0.0002, 
        lr_c = 0.0002, 
        epochs = 20, 
        training_batch = 1024,    # 5120(racecar)  # 1024 (kuka), 512
        buffer_capacity = 20000,    # 50k (racecar)  # 20K (kuka)
        batch_size = 128,  # 512 (racecar) #   128 (kuka)
        epsilon = 0.2,  # 0.07      # Clip factor required in PPO
        gamma = 0.993,  # 0.99      # discounted factor
        lmbda = 0.7,  # 0.9         # required for GAE in PPO
        tau = 0.995,                # polyak averaging factor
        alpha = 0.2,                # Entropy Coefficient   required in SAC
        use_attention = {'type': 'luong',   # type: luong, bahdanau
                        'arch': 0,
                        'return_scores' : False},        # arch: 0, 1, 2, 3
        #use_attention = None, 
        algo = 'ipg_her',               # choices: ppo, sac, ipg, sac_her, ipg_her
        env_name = 'kuka',          # environment name
        her_strategy = 'future',        # HER strategy: final, future, success 
    )

    seasons = 35
    success_value = None
    load_path = '/home/swagat/GIT/RL-Projects-SK/src/kukacam/log/kuka/ipg_her/20210911-004603/final/'
    save_path = './gradcam/'

    os.makedirs(save_path, exist_ok=True)

    # start open/AI GYM environment
    env = KukaDiverseObjectEnv(renders=False,   # True for testing
                                isDiscrete=False,
                                maxSteps=20,
                                removeHeightHack=False)

    upper_bound = env.action_space.high
    state_size = env.observation_space.shape  # (48, 48, 3)
    action_size = env.action_space.shape  # (3,)


    agent = IPGHERAgent(env, seasons, success_value, 
                        config_dict['epochs'],
                        config_dict['training_batch'],
                        config_dict['batch_size'],
                        config_dict['buffer_capacity'],
                        config_dict['lr_a'],
                        config_dict['lr_c'],
                        config_dict['gamma'],
                        config_dict['epsilon'],
                        config_dict['lmbda'],
                        config_dict['her_strategy'],
                        config_dict['use_attention'],
                        filename=None, 
                        wb_log=False,  
                        chkpt_freq=None,
                        path=None)


    # load model weights
    agent.load_model(load_path)


    # print the layers
    print(agent.feature.model.summary())

    for e in range(2):
        # reset the environment
        obs = env.reset()
        state = np.asarray(obs, dtype=np.float32) / 255.0
        goal = np.asarray(env.reset(), dtype=np.float32) / 255.0
        done = False
        t = 0
        while not done:
            action = agent.policy(state, goal)
            next_obs, reward, done, _ = agent.env.step(action)
            next_state = np.asarray(next_obs, dtype=np.float32) / 255.0

            if config_dict['use_attention']['return_scores']:
                attn_scores = np.squeeze(agent.get_attention_scores(state))
                

                fig2, axes2 = plt.subplots(1,6)
                for i in range(6):
                    axes2[i].imshow(attn_scores[:,:,i])
                    axes2[i].axis('off')
                fig2.tight_layout()
                plt.savefig(save_path+'attn_score_{}_{}.jpg'.format(e,t))

            else:

                # generate heatmap
                #heatmap = make_gradcam_heatmap(state, agent.feature.model, 'attention')
                heatmap = grad_cam2(state, agent.feature.model, agent.actor.model, 'attention', 'feature_net')
                new_img_array = save_and_display_gradcam(state, heatmap, cam_path=save_path+'cam_{}_{}.jpg'.format(e, t))
                fig, axes = plt.subplots(2, 1)
                # axes[0].imshow(obs)
                # axes[0].axis('off')
                # axes[0].set_title('Original')
                axes[0].matshow(heatmap)
                axes[0].axis('off')
                #axes[0].set_title('Heatmap')
                axes[1].imshow(new_img_array)
                axes[1].axis('off')
                #axes[1].set_title('Superimposed')
                axes[1].axis('off')
                fig.tight_layout()
                plt.savefig(save_path+'comb_{}_{}.jpg'.format(e, t))

            #plt.show()

            state = next_state
            obs = next_obs
            t += 1












