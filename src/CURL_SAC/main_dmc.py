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
import datetime
import numpy as np
from rich.console import Console

# Add the current folder to python's import path
import sys

from curl_utils import FrameStackWrapper4DMC
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir)) # parent director


# Local imports
#from  b.curl_sac_dir.curl_sac_2 import CurlSacAgent
from curl_sac_dmc import CurlSacAgent
from common.utils import check_gpu_availability, set_seed_everywhere
from curl_utils import FrameStackWrapper4DMC
########################################
# check tensorflow version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
#######################################
# Random seed
#######################################################
gpu_status = check_gpu_availability()
if gpu_status == False:
    raise SystemError('GPU device not found')
##################################### 
# Configurations & Hyper-parameters
# 
##########################################
console = Console()
config = dict(
    buffer_capacity = 50000,    # 50k (racecar)  # 20K (kuka)
    batch_size = 256,  
    use_attention = None, 
    algo = 'curl_sac',               
    env_name = 'cheetah',          # environment name
    task_name = 'run',             # task name
    image_obsvn=True,
    stack_size=3,
    include_reconst_loss=False,
    include_constcy_loss=False,
    seed=-1,
    pre_transform_image_size=100,
    aug_image_size=84,
    latent_feature_dim=50,
    alpha_c=1.0,
    alpha_r=0.0,
    alpha_cy=0.0,
)
#######################
WB_LOG = True
###########################################
# wandb related configuration
if WB_LOG:
    import wandb
    #WANDB_API_KEY=os.environ['MY_WANDB_API_KEY']
    print("WandB version", wandb.__version__)
    wandb.login()
    wandb.init(project=config['env_name'], config=config)
#######################################################33
if __name__ == "__main__":

    if config["seed"] == -1:
        config["seed"] = np.random.randint(0, 100000)
        console.log("random seed value: ", config['seed'])
        set_seed_everywhere(config['seed'])

    env = dmc2gym.make(
        domain_name=config['env_name'],
        task_name=config['task_name'],
        seed=config['seed'],
        visualize_reward=False,
        from_pixels=True,
        height=config['pre_transform_image_size'],
        width=config['pre_transform_image_size'],
        frame_skip=8
    )
    org_obs_shape = env.observation_space.shape # channel first format

     # Apply stacking wrapper to the environment
    env = FrameStackWrapper4DMC(env, config['stack_size'])

    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    upper_bound = env.action_space.high


    print('State size:', state_size)
    print('Action size:', action_size)
    print('Action upper bound:', upper_bound)

    ans = input('Do you want to continue?(Y/N)')
    if ans.lower() == 'n':
        exit()


    ######################################################
    if config['algo'] == 'curl_sac': # curl_sac
        agent = CurlSacAgent(
            state_size=state_size, 
            action_size=action_size,
            action_upper_bound=upper_bound,
            latent_feature_dim=config['latent_feature_dim'],
            buffer_capacity=config['buffer_capacity'],
            batch_size=config['batch_size'],
            include_reconst_loss=config['include_reconst_loss'],
            include_consistency_loss=config['include_constcy_loss'],
            cropped_img_size=config['aug_image_size'],
            stack_size=config['stack_size'],
            alpha_r=config['alpha_r'],
            alpha_c=config['alpha_c'],
            alpha_cy=config['alpha_cy'], 
        )

        # Train
        agent.run(env, WB_LOG=WB_LOG)


    # test
    # print('Mean Episodic Reward:', agent.validate(env, max_eps=50, render=True, load_path='./best_model/'))
    


