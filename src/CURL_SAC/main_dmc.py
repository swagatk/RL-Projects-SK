""" 
Environment: Deep Mind Control Environment (DMC) 
Algorithm: CURL-SAC

Updates:
14/03/2022: Work in progress
18/06/2022: use common encoder for actor & critic
29/07/2022: Incorporate Reconstruction loss in CURL
29/07/2022: Train DMC-Cheetah environment.

"""
# Imports
import dmc2gym
import tensorflow as tf
import pybullet_multigoal_gym as pmg
from packaging import version
import os
import sys
import datetime
import numpy as np

# Add the current folder to python's import path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir)) # parent director


# Local imports
#from  b.curl_sac_dir.curl_sac_2 import CurlSacAgent
from CURL_SAC.curl_sac import CurlSacAgent
from common.utils import check_gpu_availability, set_seed_everywhere
from curl_utils import FrameStackWrapper4DMC, Config
from config import CFG
########################################
# check tensorflow version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
#######################################
#######################################################
gpu_status = check_gpu_availability()
if gpu_status == False:
    raise SystemError('GPU device not found')
##################################### 
# Configurations & Hyper-parameters
config = Config.from_json(CFG)
##########################################
# config = dict(
#     buffer_capacity = 10000,    # 50k (racecar)  # 20K (kuka)
#     batch_size = 128,  
#     use_attention = None, 
#     algo = 'curl_sac',               
#     env_name = 'cheetah',          # environment name
#     task_name = 'run',             # task name
#     image_obsvn=True,
#     stack_size=3,
#     include_reconst_loss=False,
#     include_constcy_loss=False,
#     seed=-1,
#     pre_transform_image_size=84,
#     aug_image_size=64,
#     latent_feature_dim=50,
#     actor_dense_layers=[256, 256],
#     critic_dense_layers=[256, 256],
#     enc_conv_layers=[32, 32],
#     enc_dense_layers=[64,],
#     alpha_c=1.0,
#     alpha_r=0.0,
#     alpha_cy=0.0,
# )
#######################
EHU_SERVER=False
if EHU_SERVER:
    os.environ['MUJOCO_GL'] = "egl"
###########################################
WB_LOG = False
# wandb related configuration
if WB_LOG:
    import wandb
    #WANDB_API_KEY=os.environ['MY_WANDB_API_KEY']
    print("WandB version", wandb.__version__)
    wandb.login()
    wandb.init(project=config.env.domain_name+'-'+config.env.task_name, config=CFG)
#######################################################33
if __name__ == "__main__":

    if config.params.seed == -1:
        config.params.__dict__["seed"] = np.random.randint(0, 100000)
        print("random seed value: ", config.params.seed)
        set_seed_everywhere(config.params.seed)

    env = dmc2gym.make(
        domain_name=config.env.domain_name,
        task_name=config.env.task_name,
        seed=config.params.seed,
        visualize_reward=False,
        from_pixels=True,
        height=config.env.pre_transform_img_size,
        width=config.env.pre_transform_img_size,
        frame_skip=config.env.frame_skip
    )

    # Apply stacking wrapper to the environment
    env = FrameStackWrapper4DMC(env, config.env.stack_size)

    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    upper_bound = env.action_space.high


    print('State size:', state_size)
    print('Action size:', action_size)
    print('Action upper bound:', upper_bound)

    # ans = input('Do you want to continue?(Y/N)')
    # if ans.lower() == 'n':
    #     exit()


    ######################################################
    if config.train.algo == 'curl_sac': # curl_sac
        agent = CurlSacAgent(
            state_size=state_size, 
            action_size=action_size,
            action_upper_bound=upper_bound,
            latent_feature_dim=config.encoder.feature_dim,
            buffer_capacity=config.replay_buffer.capacity,
            batch_size=config.train.batch_size,
            alpha=config.sac.alpha,
            gamma=config.sac.gamma,
            polyak=config.sac.tau,      # polyak averaging factor
            lr=config.actor.lr,      # same learning rate for all modules
            eval_freq=config.eval.freq,
            eval_episodes=config.eval.num_episodes,
            init_steps=config.train.init_steps,
            max_training_steps=config.train.num_train_steps, 
            include_reconst_loss=config.params.include_reconst_loss,
            include_consistency_loss=config.params.include_constcy_loss,
            cropped_img_size=config.env.post_transform_img_size,
            target_update_freq=config.train.target_update_freq,
            ac_train_freq=config.train.ac_update_freq,
            enc_train_freq=config.train.enc_update_freq,
            stack_size=config.env.stack_size,
            alpha_r=config.params.alpha_r,
            alpha_c=config.params.alpha_c,
            alpha_cy=config.params.alpha_cy, 
            actor_dense_layers=config.actor.dense_layers,
            critic_dense_layers=config.critic.dense_layers,
            enc_conv_layers=config.encoder.conv_layers,
            enc_dense_layers=config.encoder.dense_layers,
        )

        # print parameters
        print(agent)

        # Train
        agent.run(env, WB_LOG=WB_LOG)


    # test
    # print('Mean Episodic Reward:', agent.validate(env, max_eps=50, render=True, load_path='./best_model/'))
    


