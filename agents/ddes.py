import random
import torch
import torch.optim as optim
import torch.nn as nn
from itertools import count
import matplotlib.pyplot as plt
from agents.base import Agent
from networks.dnn import DQN_Conv2D
from networks.bnn import DQN_VBConv2D
from utils.transition import Transition
import numpy as np

class DDESAgent(Agent):
    def __init__(self, env, is_deterministic, linear_layer_class=None, conv_layer_class=None, expl_params=None, step_params=None):
        super(DDESAgent, self).__init__(env=env, is_deterministic=is_deterministic, expl_params=expl_params, step_params=step_params)
        self.name = "Double Expected Sarsa Agent"
        self.init_message()
        
        if is_deterministic:
            self.name += " Deterministic"
            self.policy_net1 = DQN_Conv2D(stack_dim=4, n_actions=env.action_space.n)
            self.policy_net2 = DQN_Conv2D(stack_dim=4, n_actions=env.action_space.n)
        else:
            if linear_layer_class is None or conv_layer_class is None:
                raise ValueError("Linear Layer and Conv Layer class should be provided for Bayesian Neural Network")
            self.name += " Bayesian"
            self.policy_net1 = DQN_VBConv2D(n_actions=env.action_space.n, conv_layer_class=conv_layer_class, linear_layer_class=linear_layer_class)
            self.policy_net2 = DQN_VBConv2D(n_actions=env.action_space.n, conv_layer_class=conv_layer_class, linear_layer_class=linear_layer_class)
        
        self.optimizer1 = optim.Adam(self.policy_net1.parameters(), lr=self.LR)
        self.optimizer2 = optim.Adam(self.policy_net2.parameters(), lr=self.LR)
        self.current_net = 1
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.get_eps()
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net1(state).max(1)[1].view(1, 1) if self.current_net == 1 else self.policy_net2(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=Agent.device, dtype=torch.long)
    
    def select_max_action(self, state):
        with torch.no_grad():
            return self.policy_net1(state).max(1)[1].view(1, 1) if self.current_net == 1 else self.policy_net2(state).max(1)[1].view(1, 1)
        
    def optimize_model(self):
        if len(self.memory) < Agent.BATCH_SIZE:
            return
        transitions = self.memory.sample_all()
        batch = Transition(*zip(*transitions))
        self.memory.reset_memory()

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=Agent.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net1(state_batch).gather(1, action_batch) if self.current_net == 1 else self.policy_net2(state_batch).gather(1, action_batch)
        new_state_action_values = torch.zeros(Agent.BATCH_SIZE, device=Agent.device)

        with torch.no_grad():
            if self.current_net == 1:
                next_state_action_values = self.policy_net2(non_final_next_states)
            else:
                next_state_action_values = self.policy_net1(non_final_next_states)
            
            max_indices = torch.argmax(next_state_action_values, dim=1)
            max_mask = torch.zeros_like(next_state_action_values, dtype=torch.bool)
            max_mask.scatter_(1, max_indices.unsqueeze(1), 1)
            max_values = torch.where(max_mask, next_state_action_values, torch.zeros_like(next_state_action_values))
            non_max_values = next_state_action_values
            eps_threshold = self.get_eps()
            new_state_action_values[non_final_mask] = torch.sum(max_values * (1 - eps_threshold) + (eps_threshold * non_max_values) / (self.env.action_space.n))

        expected_state_action_values = (new_state_action_values * Agent.GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        if self.current_net == 1:
            self.optimizer1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net1.parameters(), 100)
            self.optimizer1.step()
            self.current_net = 2
        else:
            self.optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net2.parameters(), 100)
            self.optimizer2.step()
            self.current_net = 1
    
    def train(self, num_episodes=100):
        plt.ioff()
        episode_rewards = []
        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=Agent.device).unsqueeze(0)
            total_reward = 0
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                total_reward += reward
                reward = torch.tensor([reward], device=Agent.device)
                done = terminated or truncated
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=Agent.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    # self.plot_durations()
                    break
            episode_rewards.append(total_reward)
        print(f'Complete training {self.name} {self.NAME_SUFFIX}')
        self.plot_performance(episode_rewards)

    def evaluate(self, num_episodes):
        plt.ioff()
        self.policy_net1.eval()
        self.policy_net2.eval()
        episode_rewards = []
        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=Agent.device).unsqueeze(0)
            total_reward = 0
            for t in count():
                action = self.select_max_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                total_reward += reward
                done = terminated or truncated
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=Agent.device).unsqueeze(0)

                state = next_state

                if done:
                    self.episode_durations.append(t + 1)
                    break
            episode_rewards.append(total_reward)
        print(f'Complete evaluation {self.name} {self.NAME_SUFFIX}')
        self.plot_performance(episode_rewards, is_evaluation=True)
