# CRC-RL Model

In this project, we extend the CURL model ([Laskin, 2020](https://proceedings.mlr.press/v119/laskin20a/laskin20a.pdf)) by proposing a new loss function, called the CRC loss, to train the RL model. 

The CRC loss function includes contrastive, reconstruction and consistency loss components. 

This requires modifying the original CURL architecture to include a decoder network and a predictor network. 

More details about the proposed architecture and the algorithm could be found in [this](https://arxiv.org/pdf/2301.13473.pdf) paper. 

The code was originally developed by Darshita Jain. Her implementation could be found [here](https://github.com/darshitajain/CRC-RL-).

My implementation differs slightly from the above implementation in the sense that, I use a slightly more generic encoder/decoder/predictor architecture where the user can customize the inner layers more easily. 

This requires the places dataset which could be downloaded from [this](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) page. 

Other dependencies for this code are as follows:

* [Deepmind control suite (DMC)](https://github.com/deepmind/dm_control/)
* [dmc2gym](https://github.com/denisyarats/dmc2gym)
* Pytorch 2.0.1 with (torchvision 0.15.2, torchaudio 2.0.2, pytorch-cuda 11.7)
* Pillow 9.4.0
* Mujoco 2.3.6
* Python 3.8.16