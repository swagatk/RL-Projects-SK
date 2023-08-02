import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from rich.console import Console
from feature_extractor import Encoder, Decoder, Predictor
import augmentations
import utils
from torchsummary import summary

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

def weight_init(m):
    """ 
    Custom weight initialization for Conv2D and Linear Layers
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)  # why? square filters, H=W ??
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """
    MLP Actor Network
    """
    def __init__(self, 
                 obs_shape, 
                 action_shape, 
                 encoder_feature_dim,
                 log_std_min,
                 log_std_max,
                 learning_rate=1e-3,
                 actor_dense_layers=[128, 64, ],
                 enc_dense_layers=[128, 64, ],
                 enc_conv_layers=[32, 32],
                 ) -> None:
        super().__init__()

        self.obs_shape = obs_shape #(c, w, h)
        self.action_shape = action_shape
        self.encoder_feature_dim = encoder_feature_dim
        self.learning_rate = learning_rate
        self.enc_conv_layers = enc_conv_layers
        self.enc_dense_layers = enc_dense_layers
        self.actor_dense_layers = actor_dense_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder = Encoder(self.obs_shape,
                                   self.encoder_feature_dim,
                                   self.enc_conv_layers,
                                   self.enc_dense_layers)

        self.mlp_layers = nn.Sequential()
        for i in range(len(self.actor_dense_layers)):
            if i == 0:  # first layer
                self.mlp_layers.append(nn.Linear(self.encoder_feature_dim, self.actor_dense_layers[i]))
                self.mlp_layers.append(nn.ReLU())
            else:
                self.mlp_layers.append(nn.Linear(self.actor_dense_layers[i-1], self.actor_dense_layers[i]))
                self.mlp_layers.append(nn.ReLU())
        self.mlp_layers.append(nn.Linear(self.actor_dense_layers[-1], 2 * self.action_shape[0]))

        # Apply custom weight initialization
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, 
                compute_log_pi=True, 
                detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.mlp_layers(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        noise = torch.randn_like(mu)
        std = log_std.exp()

        pi = mu + noise * std if compute_pi else None 

        log_pi = gaussian_logprob(noise, log_std) if compute_log_pi else None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std, obs 


class QFunction(nn.Module):
    """MLP for Q function"""
    def __init__(self, obs_dim, action_dim, 
                 dense_layers=[32, 32]) -> None:
        super().__init__()

        self.input_dim = obs_dim + action_dim
        self.dense_layers = dense_layers
        self.q_func = nn.Sequential()
        for i in range(len(self.dense_layers)):
            if i == 0: # input layer
                self.q_func.append(nn.Linear(self.input_dim, self.dense_layers[i]))
            else: 
                self.q_func.append(nn.Linear(self.dense_layers[i-1], self.dense_layers[i]))
            self.q_func.append(nn.ReLU())   # inner layers have relu activation
        self.q_func.append(nn.Linear(self.dense_layers[-1], 1)) # last layer is linear

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0), "obs & action should have same number of rows"
        obs_action = torch.cat([obs, action], dim=1)
        q_value = self.q_func(obs_action)
        return q_value  
    

class Critic(nn.Module):
    """ Critic uses two q-functions """
    def __init__(self,
                 obs_shape,
                 action_shape,
                 encoder_feature_dim,
                 critic_dense_layers=[128, 64, ],
                 enc_dense_layers=[32, 32, ],
                 enc_conv_layers=[64, 64],
                 ) -> None:
        super().__init__()

        self.encoder = Encoder(obs_shape, 
                               encoder_feature_dim, 
                               enc_conv_layers,
                               enc_dense_layers)
        
        self.Q1 = QFunction(self.encoder.feature_dim, action_shape[0],
                            critic_dense_layers)
        
        self.Q2 = QFunction(self.encoder.feature_dim, action_shape[0],
                            critic_dense_layers)

        # apply custom weight initialization to parameters
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propagation to the encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        
        q2 = self.Q2(obs, action)

        return q1, q2
    

#############
## CURL 
##############
class CURL(nn.Module):
    def __init__(self, 
                 z_dim,  
                 critic,
                 target_critic,
                 learning_rate=1e-3,) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.encoder = critic.encoder
        self.target_encoder = target_critic.encoder 
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))

    def encode(self, x, detach=False, ema=False):
        '''
        Encoder: z_t = enc(x_t)
        :param x: input observation
        : return z_t
        '''
        if ema: #exponential moving average
            with torch.no_grad():
                z_out = self.target_encoder(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()

        return z_out
    
    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL
        - Compute (B, B) matrix = z_a * W_z * z_pos.T
        - positives are diagonal elements
        - negatives are all other elements
        - to compute loss, use multi-class cross-entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim , B)
        logits = torch.matmul(z_a, Wz)      # (B, B)
        logits = logits - torch.max(logits, 1)[0][:, None] 
        return logits 
    

class CurlSacAgent(object):
    ''' CURL RL Learning Agent'''
    def __init__(self, 
                 obs_shape,
                 action_shape,
                 device,
                 WB_LOG,
                 discount=0.99,
                 init_temp=0.01,
                 actor_lr=1e-3,
                 actor_beta=0.9,
                 actor_log_std_min=-10,
                 actor_log_std_max=2,
                 actor_update_freq=2,
                 actor_dense_layers=[128, 64,],
                 critic_dense_layers=[128, 64],
                 critic_target_update_freq=2,
                 critic_lr=1e-3,
                 critic_beta=0.9,
                 critic_tau=0.005,
                 encoder_feature_dim=50,
                 enc_dense_layers=[128, 64,],
                 enc_conv_layers=[64, 32, ],
                 encoder_lr=1e-3,
                 decoder_lr=1e-3,
                 encoder_tau=0.005,
                 curl_encoder_update_freq=1,
                 alpha_lr=1e-3,
                 alpha_beta=0.9,
                 decoder_weight_lambda=1e-7,
                 decoder_latent_lambda=1e-6,
                 detach_encoder=False,
                 predictor_dense_layers=[64, 64, ],
                 c1=0.33,
                 c2=0.33,
                 c3=0.33,
                 log_interval=100,
                 ) -> None:
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.target_critic_update_freq = critic_target_update_freq
        self.curl_encoder_update_freq = curl_encoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.image_size = obs_shape[-1]  # h of (c, w, h); assume w=h
        self.detach_encoder = detach_encoder
        self.log_interval = log_interval
        self.c1 = c1    # weightage for contrastive loss
        self.c2 = c2    # weightage for reconstruction loss
        self.c3 = c3    # weightage for consistency loss
        self.WB_LOG = WB_LOG


        self.actor = Actor(
            obs_shape,
            action_shape,
            encoder_feature_dim,
            actor_log_std_min,
            actor_log_std_max,
            actor_lr,
            actor_dense_layers,
            enc_dense_layers,
            enc_conv_layers
        ).to(device)

        print("Actor model:")
        summary(self.actor, obs_shape)


        self.critic = Critic(
            obs_shape,
            action_shape,
            encoder_feature_dim,
            critic_dense_layers,
            enc_dense_layers,
            enc_conv_layers
        ).to(device)

        print("Critic Model:")
        summary(self.critic, [obs_shape, action_shape])

        self.target_critic = Critic(
            obs_shape,
            action_shape,
            encoder_feature_dim,
            critic_dense_layers,
            enc_dense_layers,
            enc_conv_layers
        ).to(device)

        # Initially copy weights between the two critics
        self.target_critic.load_state_dict(self.critic.state_dict())

        # tie encoder weights between actor, critic & curl
        self.actor.encoder.copy_model_params(self.critic.encoder)


        # set the entropy coefficient as a tunable parameter.
        self.log_alpha = torch.tensor(np.log(init_temp)).to(device)
        self.log_alpha.requires_grad = True 

        self.target_entropy = -np.prod(action_shape)

        # shape of conv layer output in the encoder
        enc_conv_out_shape = self.actor.encoder.get_conv_out_shape()

        self.decoder = Decoder(obs_shape, 
                               encoder_feature_dim,
                               enc_conv_layers[::-1],
                               enc_dense_layers[::-1],
                               enc_conv_out_shape
        ).to(device)

        print("Decoder Model:")
        summary(self.decoder, (encoder_feature_dim, ))

        # Predictor for Consistency Loss
        self.predictor = Predictor(
            self.critic.encoder,
            predictor_dense_layers
        ).to(device)

        print("Predictor Model:")
        summary(self.predictor, obs_shape)

        # CURL 
        self.curl = CURL(encoder_feature_dim,
                         self.critic, 
                         self.target_critic,
                         ).to(device)
        
            
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=actor_lr,
            betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            betas=(critic_beta, 0.999)
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], 
            lr=alpha_lr,
            betas=(alpha_beta, 0.999) 
        )

        self.curl_optimizer = torch.optim.Adam(
            self.curl.parameters(), 
            lr=encoder_lr
        )

        self.pred_optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=encoder_lr
        )

        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(),
            lr=encoder_lr
        )
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # toggle training attribute of parameters.
        self.train()
        self.target_critic.train()

    def train(self, training=True):
        '''
        function to toggle between train and eval mode.
        '''
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.curl.train(training)
        if self.decoder is not None:
            self.decoder.train(training)
        if self.predictor is not None:
            self.predictor.train(training)

    
    @property 
    def alpha(self):
        return self.log_alpha.exp()
    

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).float()
            obs = obs.unsqueeze(0) # (-1, c, h, w)
            mu, _, _, _, _ = self.actor(obs, 
                                        compute_pi=False,
                                        compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()
        
    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = augmentations.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).float()
            obs = obs.unsqueeze(0) # (-1, c, h, w)
            _, pi, _, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()
        
    
    def update_critic(self, obs, action, reward, 
                      next_obs, not_done, step):
        with torch.no_grad():
            _, policy_action, log_pi, _, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.target_critic(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        
        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder
        )

        # Loss function for training critic
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                F.mse_loss(current_Q2, target_Q)
        
        if step % self.log_interval == 0 and self.WB_LOG:
            wandb.log({"train_critic/loss": critic_loss, "step": step })

        
        # train critic
        self.critic_optimizer.zero_grad() # clear gradients
        critic_loss.backward()  # compute gradients of loss wrt parameters
        self.critic_optimizer.step()    # update parameters


    def update_actor_and_alpha(self, obs, step):

        # detach encoder, so we don't update it with the actor loss

        _, pi, log_pi, log_std, _ = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)


        # train the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Log
        if step % self.log_interval == 0 and self.WB_LOG:
            wandb.log({"train_actor/loss": actor_loss, "step": step})
            wandb.log({"train_actor/target_entropy":self.target_entropy, "step":step})
            wandb.log({"train_actor/entropy": entropy.mean(), "step" : step})
            wandb.log({"train_alpha/loss": alpha_loss, "step" : step})
            wandb.log({"train_alpha/alpha": self.alpha, "step": step})


    def update_curl_encoder_and_decoder(self, org_obs, obs_anchor, obs_pos, 
                                 target_obs, step):
        
        # contrastive loss 
        z_a = self.curl.encode(obs_anchor)
        z_pos = self.curl.encode(obs_pos)
        logits = self.curl.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        curl_loss = self.cross_entropy_loss(logits, labels)

        # reconstruction loss
        h = self.critic.encoder(obs_anchor)
        if target_obs.dim() == 4: # (-1, c, h, w)
            # preprocess images to be in range [-0.5, 0.5]
            target_obs = utils.preprocess_obs(target_obs)

        rec_obs = self.decoder(h)

        if target_obs.size() == rec_obs.size():
            rec_loss = F.mse_loss(target_obs, rec_obs)
        else:
            print("The shape of reconstructed observation must match with that of original observation for computing reconstruction loss.")
            exit(1)


        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        # consistency loss
        h0 = self.predictor(obs_anchor)
        with torch.no_grad():
            h1 = self.target_critic.encoder(org_obs)
        h0 = F.normalize(h0, p=2, dim=1)
        h1 = F.normalize(h1, p=2, dim=1)

        consis_loss = F.mse_loss(h0, h1)

        # CRC loss 
        loss = self.c1 * curl_loss + self.c2 * rec_loss + self.c3 * consis_loss + self.decoder_latent_lambda * latent_loss 


        # update encoder & decoder parameters
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.curl_optimizer.zero_grad()
        self.pred_optimizer.zero_grad()

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.curl_optimizer.step()
        self.pred_optimizer.step()

        if step % self.log_interval == 0 and self.WB_LOG:
            wandb.log({"train/curl_loss": curl_loss, "step": step})
            wandb.log({"train/rec_loss":rec_loss, "step": step})
            wandb.log({"train/consy_loss": consis_loss, "step": step})
            wandb.log({"train/total_loss": loss, "step": step})

    
    def update(self, replay_buffer, step):
        
        # sample a batch from replay buffer
        (
            orig_obs, 
            obs,
            action,
            reward,
            next_obs,
            not_done,
            info_dict
        ) = replay_buffer.sample_img_obs()

        # update critic at every step
        self.update_critic(obs, action, reward, next_obs, not_done, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, step)

        if step % self.target_critic_update_freq == 0:
            utils.soft_update_params(self.critic.Q1, self.target_critic.Q1, self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.target_critic.Q2, self.critic_tau)
            utils.soft_update_params(self.critic.encoder, self.target_critic.encoder, self.critic_tau)


        if step % self.curl_encoder_update_freq == 0:
            obs_anchor, obs_pos = info_dict["obs_anchor"], info_dict["obs_pos"]
            self.update_curl_encoder_and_decoder(
                orig_obs, obs_anchor, obs_pos, obs_anchor, step
            )


        

        










