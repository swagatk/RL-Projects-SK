import gym
import torch as th

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.space)
    :param features_dim: (int) number of features extracted. This corresponds to
          the number of unit for the last layer
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first format)
        n_input_channels = observation_space.shape[0]
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2, padding=0),
            th.nn.ReLU(),
            th.nn.BatchNorm2d(16),
            th.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            th.nn.ReLU(),
            th.nn.BatchNorm2d(32),
            th.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            th.nn.ReLU(),
            th.nn.BatchNorm2d(64),
            th.nn.Flatten()
        )

        # compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = th.nn.Sequential(
            th.nn.Linear(n_flatten, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, features_dim),
            th.nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


