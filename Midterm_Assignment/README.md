# A2C Reinforcement Learning Implementation using Pytorch

Here, we have used used Advantage Actor Critic (A2C) to solve BipedalWalker-v2 OpenAI gym environment.

## Codes & Results

* [plotting.py](/plotting.py): contains plotting function plot_episode_stats
* [a2cpytorch.py](/a2cpytorch.py) : contains A2C neural network architecture with training function
* Training the algorithm for 20000-50000 episodes solves the environment

## To Run

* Run 
```bash
  python a2cpytorch.py -g [gamma] -ep [number_of_episodes]
``` 
* Always enter both the command line arguments

## Dependencies

* pytorch >= 0.4.0
* numpy
* gym
* matplotlib
* pandas
