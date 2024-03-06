# Solving Mountain Car problem

## DQN / DDQN / PER

DQN/DDQN with and without priority experience replay (PER) solves the problem in a few episodes. It can run both on GPU as well as CPU. It takes only a few minutes to witness success. It is a success if the car can reach the top of the hill (x = 0.5) in less than 200 steps. 

### Install Dependencies

```
pip install gymnasium
pip install gymnasium[classic-control]
```

### Relevant files

*  `./dqn.py`
* `./train.py`
* `../common/buffer.py`
* `../common/sumtree.py`

### Output
The output looks something like this:

```
e:10, episodic reward: 41.27, avg ep reward: 21.62, epsilon: 0.00
 Successfully solved the problem in 11 epsisodes,                                 max_pos: 0.53, steps: 117

e:11, episodic reward: 43.26, avg ep reward: 23.42, epsilon: 0.00
 Successfully solved the problem in 12 epsisodes,                                 max_pos: 0.50, steps: 181

e:12, episodic reward: 49.51, avg ep reward: 25.43, epsilon: 0.00
 Successfully solved the problem in 13 epsisodes,                                 max_pos: 0.51, steps: 156

e:13, episodic reward: 47.55, avg ep reward: 27.01, epsilon: 0.00
 Successfully solved the problem in 14 epsisodes,                                 max_pos: 0.52, steps: 157

```

![GIF Animation](../../dqn_mc.gif)
## Execution 

`$python3 train.py`

## generate GIF animation

Pass one of the saved models to the  `validate()` function to generate GIF animation for the trained model. You need to comment/uncomment necessary lines in the `train.py` file and execute the above command