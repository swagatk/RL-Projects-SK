# Single-stage manipulation environments
# Reach, Push, PickAndPlace, Slide
import pybullet_multigoal_gym as pmg
# Install matplotlib if you want to use imshow to view the goal images
import matplotlib.pyplot as plt
import numpy as np

camera_setup = [
    {
        'cameraEyePosition': [-1.0, 0.25, 0.6],
        'cameraTargetPosition': [-0.6, 0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 128,
        'render_height': 128
    },
    {
        'cameraEyePosition': [-1.0, -0.25, 0.6],
        'cameraTargetPosition': [-0.6, -0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 128,
        'render_height': 128
    }
]

env = pmg.make_env(
    # task args ['reach', 'push', 'slide', 'pick_and_place', 
    #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
    task='reach',
    gripper='parallel_jaw',
    num_block=4,  # only meaningful for multi-block tasks
    render=False,
    binary_reward=True,
    max_episode_steps=5,
    # image observation args
    image_observation=True,
    depth_image=False,
    goal_image=True,
    visualize_target=True,
    camera_setup=camera_setup,
    observation_cam_id=0,
    goal_cam_id=1,
    # curriculum args
    use_curriculum=True,
    num_goals_to_generate=90)

fig, axes = plt.subplots(1,3)
obs = env.reset()
print('Observation space:', env.observation_space)
print('Action Space:', env.action_space)
print('Shape of action space:', env.action_space.__getattribute__('shape'))
#print('Shape of Observation space:', env.observation_space['observation'].__getattribute__('shape'))
print('Reward Range:',env.reward_range)

input('Press Enter to continue ...')
t = 0
ep_reward = 0
while True:
    t += 1
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print('state: ', obs['state'], '\n',
          'desired_goal: ', obs['desired_goal'], '\n',
          'achieved_goal: ', obs['achieved_goal'], '\n',
          'reward: ', reward, '\n',
          'action:', action, '\n')
    ep_reward += reward
    axes[0].imshow(obs['observation'])
    axes[0].set_title('Observation')
    axes[1].imshow(obs['desired_goal_img'])
    axes[1].set_title('desired_goal')
    axes[2].imshow(obs['achieved_goal_img'])
    axes[2].set_title('achieved_goal')
    plt.pause(0.00001)      
    if done:
        print('Max steps: {}, Total episodic reward:{}'.format(t, ep_reward))
        env.reset()
        t = 0
        ep_reward = 0