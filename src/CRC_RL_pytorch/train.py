import numpy as np
import torch
import os 
import time
import wandb 
import utils
from rich.console import Console 
from config import CFG 
from curl_sac import CurlSacAgent 
from utils import Config 
from augmentations import center_crop_image, random_crop, random_overlay, random_conv
from collections import Counter 
from wrappers import make_env 

EHU = False
if EHU:
    os.environ['MUJOCO_GL'] = "egl"


config = Config.from_json(CFG)
WB_LOG = True 
console = Console()

transforms = {
    "random_crop": random_crop,
    "random_conv": random_conv,
    "center_crop_image": center_crop_image,
    "random_overlay": random_overlay,
}

if config.params.seed == -1:
    config.params.__dict__["seed"] = np.random.randint(1, 1000000)
    console.log("random seed value", config.params.seed)
utils.set_seed_everywhere(config.params.seed)


def make_agent(obs_shape, action_shape, config, device, WB_LOG):
    if config.train.agent == "curl_sac":
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            WB_LOG=WB_LOG,
            encoder_feature_dim=config.encoder.feature_dim,
            discount=config.sac.discount,
            init_temp=config.sac.init_temp,
            alpha_lr=config.sac.alpha_lr,
            alpha_beta=config.sac.alpha_beta,
            actor_lr=config.actor.lr,
            actor_beta=config.actor.beta,
            actor_log_std_min=config.actor.log_std_min,
            actor_log_std_max=config.actor.log_std_max,
            actor_update_freq=config.actor.update_freq,
            actor_dense_layers=config.actor.dense_layers,
            critic_lr=config.critic.lr,
            critic_beta=config.critic.beta,
            critic_tau=config.critic.tau,
            critic_target_update_freq=config.critic.target_update_freq,
            encoder_lr=config.encoder.lr,
            encoder_tau=config.encoder.tau,
            enc_conv_layers=config.encoder.conv_layers,
            enc_dense_layers=config.encoder.dense_layers,
            decoder_lr=config.decoder.lr,
            decoder_latent_lambda=config.decoder.latent_lambda, ## what is this?
            decoder_weight_lambda=config.decoder.weight_lambda,  ## ??
            log_interval=config.params.log_interval,
            detach_encoder=config.params.detach_encoder,
            predictor_dense_layers=config.predictor.dense_layers,
            c1=config.params.c1,
            c2=config.params.c2,
            c3=config.params.c3
        )
    else:
        assert f"agent is not supported: {config.train.agent}"


def evaluate(env_val, agent, num_episodes, step, device):
    all_ep_rewards = []
    def run_eval_loop(sample_stochastically=True):
        for i in range(num_episodes):
            obs = env_val.reset()
            done = False
            episode_reward = 0
            while not done:
                obs = transforms["center_crop_image"](obs, config.env.image_size)
                obs = torch.as_tensor(obs, device=device).float()
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                
                obs, reward, done, _ = env_val.step(action)
                episode_reward += reward
        
            all_ep_rewards.append(episode_reward)
        mean_ep_reward = np.mean(all_ep_rewards)

        if WB_LOG:
            wandb.log({"eval/mean_episode_reward": mean_ep_reward, "step": step})

    run_eval_loop(sample_stochastically=False)


def train(env, env_val, agent, replay_buffer, device):
    episode, ep_reward, done = 0, 0, True 
    all_ep_reward = []
    start_time = time.time()
    for step in range(config.train.num_train_steps):

        # evaluate agent periodically
        if step > 0 and step % config.eval.eval_freq == 0:
            evaluate(env_val, agent, config.eval.num_eval_episodes, step, device)

        
        if done:
            all_ep_reward.append(ep_reward)
            ep_duration = time.time() - start_time

            if step % config.params.log_interval == 0 and WB_LOG:
                wandb.log({"train/episode_reward": ep_reward, "step": step})
                wandb.log({"train/mean_ep_reward": np.mean(all_ep_reward), "step": step})

            console.log(
                f"Train | Episode - {episode}, Step - {step}, Episode Reward - {ep_reward:.4f}, Duration - {ep_duration:.4f}"
            )

            obs = env.reset()
            done = False
            ep_reward = 0
            episode += 1
            start_time = time.time()


        # sample action 
        if step < config.train.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        next_obs, reward, done, _ = env.step(action)

        done_bool = 0 if episode + 1 == env._max_episode_steps else float(done)
        ep_reward += reward 
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        # run training updates
        if step >= config.train.init_steps:
            for _ in range(config.train.num_updates):
                agent.update(replay_buffer, step)

        # prepare for next iteration
        obs = next_obs

def main():

    if WB_LOG:
        wandb.login()
        wandb.init(project=config.env.domain_name, config=CFG)

    if config.env.transform2 == 'random_conv':
        pre_transform_image_size = config.env.image_size
    else:
        pre_transform_image_size = config.env.pre_transform_image_size

    env = make_env(
        domain_name=config.env.domain_name, 
        task_name=config.env.task_name,
        seed=config.params.seed,
        episode_length=config.env.episode_length,
        frame_stack=config.env.frame_stack,
        action_repeat=config.env.action_repeat,
        image_size=config.env.pre_transform_image_size,
        mode='train'
    )
    env.seed(config.params.seed)

    test_env = make_env(
        domain_name=config.env.domain_name,
        task_name=config.env.task_name,
        seed=config.params.seed,
        episode_length=config.env.episode_length,
        frame_stack=config.env.frame_stack,
        action_repeat=config.env.action_repeat,
        image_size=config.env.eval_pre_transform_image_size,
        mode=config.eval.mode
    ) if config.eval.mode is not None else None

    test_env.seed(config.params.seed)

    action_shape = env.action_space.shape
    obs_shape = (
        3 * config.env.frame_stack,
        config.env.image_size,
        config.env.image_size
    )

    pre_aug_obs_shape = (
        3 * config.env.frame_stack,
        config.env.pre_transform_image_size,
        config.env.pre_transform_image_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=config.replay_buffer.capacity,
        batch_size=config.train.batch_size,
        device=device,
        image_size=config.env.image_size,
        transform1=transforms[config.env.transform1],
        transform2=transforms[config.env.transform2]
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        config=config,
        device=device,
        WB_LOG=WB_LOG
    )

    train(env, test_env, agent, replay_buffer, device)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
        


        








        


