import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchsummary import summary 
import numpy as np



def copy_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias
    #print("Check if they have same ids")
    #print(id(trg.weight) == id(src.weight))


#########
## Encoder
###################
class Encoder(nn.Module):
    def __init__(self,
    obs_shape,
    feature_dim,
    conv_layers = [64, 32, 32],
    dense_layers = [1028, 512, 256]) -> None:
        super().__init__()
        assert len(obs_shape) == 3, "image observation of shape (c, w, h) is expected"
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.conv_layers = conv_layers 
        self.dense_layers = dense_layers 
        self.conv_out_shape = None 


        self.convs = nn.Sequential()
        for i in range(len(self.conv_layers)):
            if i == 0:
                self.convs.append(nn.Conv2d(self.obs_shape[0], self.conv_layers[i], 
                                                 kernel_size=3, stride=1))
            else:
                self.convs.append(nn.Conv2d(self.conv_layers[i-1], self.conv_layers[i],
                                                 kernel_size=3, stride=1))
            self.convs.append(nn.ReLU())
            self.convs.append(nn.MaxPool2d(kernel_size=2))


        self.fcs = nn.Sequential()
        for i in range(len(self.dense_layers)):
            if i == 0:
                self.fcs.append(nn.LazyLinear(self.dense_layers[i]))
            else:
                self.fcs.append(nn.Linear(self.dense_layers[i-1], self.dense_layers[i]))
            self.fcs.append(nn.ReLU())
        self.fcs.append(nn.Linear(self.dense_layers[-1], self.feature_dim))


        

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, obs, detach=False):
        obs = obs / 255.0

        out = self.convs(obs)
        out = out.view(obs.shape[0], -1)
        out = self.fcs(out)

        if detach:
            out = out.detach()

        return out

    def copy_model_params(self, source):
        assert(type(source) == type(self)), "Both models should be of same type"
        self.load_state_dict(source.state_dict())




############
## Decoder
#####################
class Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, 
                 conv_layers=[32, 32, 64], 
                 dense_layers=[256, 512, 1028]) -> None:
        super().__init__()

        self.obs_shape = obs_shape
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.feature_dim = feature_dim

        self.fcs = nn.Sequential()

        # dense layers
        for i in range(len(self.dense_layers)):
            if i == 0:
                self.fcs.append(nn.Linear(self.feature_dim, self.dense_layers[i]))
            else:
                self.fcs.append(nn.Linear(self.dense_layers[i-1], self.dense_layers[i]))
            self.fcs.append(nn.ReLU())
        self.fcs.append(nn.Linear(self.dense_layers[-1], np.prod(self.obs_shape)))
        self.fcs.append(nn.ReLU())

        # Unflatten layer
        #self.model.append(nn.Unflatten(self.obs_shape))

        self.deconvs = nn.Sequential()
        # Deconvolution
        for i in range(len(self.conv_layers)):
            if i == 0:
                self.deconvs.append(nn.ConvTranspose2d(self.obs_shape[0], self.conv_layers[i], 
                                                       kernel_size=3, stride=1, padding=1))
            elif i < len(self.conv_layers) - 1:
                self.deconvs.append(nn.ConvTranspose2d(self.conv_layers[i-1], self.conv_layers[i], 
                                                       kernel_size=3, stride=1, padding=1))
            else:   # last layer
                self.deconvs.append(nn.ConvTranspose2d(self.conv_layers[i], self.obs_shape[0], 
                                                       kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.fcs(x)
        x = x.view((-1,) + self.obs_shape)
        x = self.deconvs(x)

        return x


#########################
## Predictor
#######################

class MLP(nn.Module):
    def __init__(self, input_dim, dense_layers:list, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential()

        for i in range(len(dense_layers)-1):
            if i == 0: # input layers
                self.mlp.append(nn.Linear(input_dim, dense_layers[i]))
            else: # hidden layers
                self.mlp.append(nn.Linear(dense_layers[i-1], dense_layers[i]))
            self.mlp.append(nn.BatchNorm1d(dense_layers[i]))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(dense_layers[-1], output_dim)) # output layer

    def forward(self, x):
        return self.mlp(x)
    

class Predictor(nn.Module):
    def __init__(self, encoder, dense_layers:list) -> None:
        super().__init__()
        self.encoder = encoder
        self.mlp = MLP(self.encoder.feature_dim, 
                       dense_layers, 
                       self.encoder.feature_dim)
        
    def forward(self, x):
        enc_x = self.encoder(x)
        return self.mlp(enc_x)






        
