"""
Author: Wenbo Duan
Date: 29th, April, 2022
Update" 9th, June, 2022
Usage: Main scripts for the reinforcement learning training with VPG algorithm
"""

import sys
"Please replace the path with the root path of the folder 'ug_project' in your computer!"
sys.path.append("/Users/duanwenbo/Documents/CodeSpace/UG_Project/ug_project")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from Environment.Channel import SISO_Channel
from Environment.utils import sum_rate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, input, hidden=128, output=2) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden,1)
        self.sigma_head = nn.Linear(hidden,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        mu = F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, input, hidden=64, output=1) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)
    
    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VPG():
    def __init__(self,EPISODE_NUM,GAMMA,LEARNING_RATE, channel, hidden_layer_actor,hidden_layer_critic, save_path):
        self.EPISODE_NUM = EPISODE_NUM
        self.GAMMA = GAMMA
        self.LEARNING_RATE =LEARNING_RATE
        self.env = channel
        self.hidden_layer_actor = hidden_layer_actor
        self.hidden_layer_critic = hidden_layer_critic
        self.actor = Actor(input=len(channel.state),
                            hidden=self.hidden_layer_actor).to(device)
        self.critic = Critic(input=len(channel.state),
                            hidden=self.hidden_layer_critic).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.LEARNING_RATE)
        self.critic_loss_func = nn.MSELoss(reduce='sum')
        self.save_path = save_path

    
    def choose_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        # perdict a normal distribution
        mu, sigma = self.actor(state)
        distribution = Normal(mu,sigma)
        action = distribution.sample()
        return action.item()

    def compute_advantage(self, rewards, state_values):
        r2g = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            r2g[i] = rewards[i] + (r2g[i+1] if i + 1 < len(rewards) else 0)
        
        advantages = r2g - np.array(state_values)

        r2g = torch.as_tensor(r2g, dtype=torch.float32).to(device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32).to(device)
        return advantages, r2g
    
    def train(self):
        for i in range(self.EPISODE_NUM):
            result = []
            for k in range(self.env.pair_num):
                states = []
                rewards = []
                actions = []
                next_states = []
                states_values = []
                # k: the index of the Transmitter(Tx)
                # i: the index of the episode
                state = self.env.reset(k)
                while True:
                    action = self.choose_action(state)
                    next_state, reward, done, _ = self.env.step(Tx_No=k, action=action)
                    states.append(state)
                    rewards.append(reward)
                    actions.append(action)
                    next_states.append(next_state)
                    states_values.append(self.critic(torch.as_tensor(state,dtype=torch.float32)).detach().item())

                    state = next_state
                    if done:
                        break
                
                # optimize actor
                states = torch.tensor(states,dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.float32).to(device)


                self.actor_optimizer.zero_grad()
                mus, sigmas = self.actor(states)
                distributions = Normal(mus,sigmas)
                # distributions = Categorical(actor(states))
                actions = torch.unsqueeze(actions, 1)
                log_probs = distributions.log_prob(actions)
                advantages = self.compute_advantage(rewards, states_values)[0]
                advantages = torch.unsqueeze(advantages, 1)
                actor_loss =  - torch.sum(torch.mul(log_probs, advantages))

                actor_loss.backward()
                self.actor_optimizer.step()

                # opyimize critic
                self.critic_optimizer.zero_grad()
                rtg = self.compute_advantage(rewards, states_values)[1]
                estimate_v = self.critic(states).squeeze()
                critic_loss = self.critic_loss_func(rtg, estimate_v)
                critic_loss.backward()
                self.critic_optimizer.step()

                sub_result = np.sum(rewards) / len(rewards)
                result.append(sub_result)

            print(result[-1], i)
        torch.save(self.actor, self.save_path)
        print("training finished!")
    
    # def validate(self):
    #     agent = torch.load(self.save_path)
    #     self.actor = agent
    #     power_ratio = []
        
    #     for k in range(self.env.pair_num):
    #         state = self.env.reset(k)
    #         print(state)
    #         action = self.choose_action(state)
    #         power_ratio.append(action)
    #     print(power_ratio)
               
    #     H = np.reshape(self.env.CSI, (self.env.pair_num, self.env.pair_num))
    #     P = np.array(power_ratio)
    #     capacity = sum_rate(H, P)
    #     return {"best_capacity":capacity,
    #         "best_power_ratio":power_ratio}

    


if __name__ == "__main__":
    channel = SISO_Channel(N_pairs=4,
                           seed=308)
                           
    vpg = VPG(EPISODE_NUM=1500,
              GAMMA=0.98,
              LEARNING_RATE=0.00001,
              channel=channel,
              hidden_layer_actor=128,
              hidden_layer_critic=64,
              save_path="agent.pt")
    vpg.train()
  
    
   
