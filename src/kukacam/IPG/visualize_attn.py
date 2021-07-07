import sys
import tensorflow as tf
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from ipg_her import IPGHERAgent
from IPG.ipg import IPGAgent



import numpy as np
import matplotlib.pyplot as plt
from common.grad_cam import save_and_display_gradcam, make_gradcam_heatmap


if __name__ == '__main__':

    # #### Hyper-parameters
    SEASONS = 100  # 35
    success_value = 70
    lr_a = 0.0002  # 0.0002
    lr_c = 0.0002  # 0.0002
    epochs = 20
    training_batch = 3072  # 3072 (racecar)  # 1024 (kuka), 512
    buffer_capacity = 50000  # 50k (racecar)  # 20K (kuka)
    batch_size = 256  # 256 (racecar) #   28 (kuka)
    epsilon = 0.2  # 0.07
    gamma = 0.993  # 0.99
    lmbda = 0.7  # 0.9
    use_mujoco = False
    use_attention = True

    # start open/AI GYM environment
    env = KukaDiverseObjectEnv(renders=False,   # True for testing
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)

    upper_bound = env.action_space.high
    state_size = env.observation_space.shape  # (48, 48, 3)
    action_size = env.action_space.shape  # (3,)

    agent = IPGHERAgent(env,
                        SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size,
                        buffer_capacity, epsilon, gamma, lmbda, use_attention,
                        use_mujoco)

    # load model weights
    agent.load_model('../trained_models', 'actor_weights.h5', 'critic_weights.h5', 'baseline_wts.h5')



    print(agent.feature.model.summary())

    # for e in range(5):
    #     # reset the environment
    #     if use_mujoco:
    #         obs = agent.env.reset()["observation"]
    #         state = obs
    #     else:
    #         obs = agent.env.reset()
    #         state = np.asarray(obs, dtype=np.float32) / 255.0
    #     done = False
    #     t = 0
    #     while not done:
    #         action = agent.policy(state)
    #         next_obs, reward, done, _ = agent.env.step(action)
    #
    #         if use_mujoco:
    #             next_state = next_obs["observation"]
    #         else:
    #             next_state = np.asarray(next_obs, dtype=np.float32) / 255.0
    #
    #         # generate heatmap
    #         heatmap = make_gradcam_heatmap(state, agent.feature.model, 'multiply_2')
    #         #heatmap = grad_cam2(state, agent.feature.model, agent.actor.model, 'attention_2', 'feature_net')
    #         new_img_array = save_and_display_gradcam(state, heatmap, cam_path='./gradcam/cam_l2_{}_{}.jpg'.format(e, t))
    #         fig, axes = plt.subplots(2, 2)
    #         axes[0][0].imshow(obs)
    #         axes[0][0].axis('off')
    #         #axes[0][0].set_title('Original')
    #         axes[0][1].matshow(heatmap)
    #         axes[0][1].axis('off')
    #         #axes[0][1].set_title('Heatmap')
    #         axes[1][0].imshow(new_img_array)
    #         axes[1][0].axis('off')
    #         #axes[1][0].set_title('Superimposed')
    #         axes[1][1].axis('off')
    #         fig.tight_layout()
    #         plt.savefig('./gradcam/comb_l2_{}_{}.jpg'.format(e, t))
    #         plt.show()
    #
    #         state = next_state
    #         obs = next_obs
    #         t += 1












