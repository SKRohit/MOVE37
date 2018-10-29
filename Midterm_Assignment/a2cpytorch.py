import argparse
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from itertools import count
from plotting import plot_episode_stats
#%matplotlib qt

parser = argparse.ArgumentParser(description='Pass value of gamma and num_episodes')
parser.add_argument('-g', '--gamma', type=float, metavar='', required=True, help='Discount Factor')
parser.add_argument('-ep', '--episodes', type=int, metavar='', required=True, help='Numner of Episodes')
args = parser.parse_args()

#initialise the environment, state_dimesions and action_dimensions
env = gym.make('BipedalWalker-v2')
N_S, N_A = env.env.observation_space.shape[0],env.env.action_space.shape[0]


#function to initialize weights
def set_init(layers):
    for layer in layers:
        nn.init.xavier_normal(layer.weight)
        nn.init.constant_(layer.bias, 0.1)


#Advantage Actor_Critic Architecture
class ActorCritic(nn.Module):

	def __init__(self,state_dim,action_dim):
		super(ActorCritic, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden1 = nn.Linear(state_dim,100)
		self.hidden2 = nn.Linear(100,100)
		self.value = nn.Linear(100,1)
		self.mu = nn.Linear(100,self.action_dim)
		self.sigma = nn.Linear(100,self.action_dim)
		set_init([self.hidden1, self.hidden2, self.mu, self.sigma])
		self.distribution = torch.distributions.Normal

	def forward(self,state):
		hidden1 = torch.relu(self.hidden1(state))
		hidden2 = torch.relu(self.hidden2(hidden1))
		value = self.value(hidden2)
		mu = 2 * torch.tanh(self.mu(hidden2))
		sigma = torch.clamp(F.softplus(self.sigma(hidden2)), 1e-5, 5)
		return mu, sigma, value

	def choose_action(self,state):
		self.training = False
		mu, sigma, _ = self.forward(state)
		m = self.distribution(mu, sigma)
		return torch.clamp(m.sample(),-1,1)

	#calculates total loss for ActorCritic network
	def loss_func(self,state, action, reward, next_state, gamma=0.98):
		self.train()
		mu, sigma, curr_value = self.forward(state)
		__, _, next_value = self.forward(next_state)
		m = self.distribution(mu, sigma)

		td_target = reward + gamma * next_value
		td_error = td_target - curr_value
		loss = nn.MSELoss()
		critic_loss = loss(curr_value, td_target.detach()) 

		log_prob = m.log_prob(action)
		actor_loss = -log_prob*td_error.detach()
		actor_loss -= 1e-1 * m.entropy().mean()
		total_loss = (critic_loss + actor_loss).mean()
		return total_loss


#initialise out advantage actor-critic model and optimizer
actor_critic = ActorCritic(N_S,N_A)
ac_optim = torch.optim.Adam(actor_critic.parameters(), lr = 0.01)


#training function
def ac_train(env,actor_critic,ac_optim,n_episodes,gamma=0.98):

	#tuples to store episode lengths and epsiode rewards
	EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"]) 
	stats = EpisodeStats(episode_lengths=np.zeros(n_episodes),episode_rewards=np.zeros(n_episodes))
	#tuples to store state, action, reward, next_state, done
	Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

	for i_episode in range(n_episodes):

		state = torch.Tensor(env.reset())
		#list to store all the steps taken during training
		steps = []  

		for t in count():
			env.render()
			action = actor_critic.choose_action(state)
			next_state, reward, done, _ = env.step(action)
			next_state = torch.Tensor(next_state)

			steps.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))
			stats.episode_rewards[i_episode] += reward

			#calculate total loss
			total_loss = actor_critic.loss_func(state, action, reward, next_state, gamma)  

			ac_optim.zero_grad()
			total_loss.backward()
			ac_optim.step()

			print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, n_episodes, stats.episode_rewards[i_episode - 1]), end="")
			if done:
				stats.episode_lengths[i_episode] = t
				break
			state = next_state
	return stats, steps

if __name__ == '__main__':

	gamma, num_episodes = args.gamma, args.episodes
	stats, steps = ac_train(env,actor_critic,ac_optim,num_episodes,gamma)

	#Plot 3 plots: episode_reward vs time, episode_length vs time, episode_number vs time
	plot_episode_stats(stats)
