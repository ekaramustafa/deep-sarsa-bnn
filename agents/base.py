import math
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.replay_memory import ReplayMemory
from networks.base import QNetwork

class Agent():
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    NAME_SUFFIX = "" # For saving results of different experiments 
    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env, is_deterministic, expl_params=None,step_params=None):
        if(expl_params is not None):
            Agent.EPS_START = expl_params.eps_start 
            Agent.EPS_END = expl_params.eps_end
            Agent.EPS_DECAY = expl_params.eps_decay
            Agent.NAME_SUFFIX = expl_params.name
        
        if(step_params is not None):
            Agent.BATCH_SIZE = step_params.batch_size
            Agent.GAMMA = step_params.gamma
            Agent.TAU = step_params.tau
            Agent.LR = step_params.lr
            Agent.NAME_SUFFIX = step_params.name # if both are provided, step_params will overwrite expl_paramss, design choice
        
        self.env = env
        self.name = "Base Agent"
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        self.is_deterministic = is_deterministic
    
    def init_message(self):
        print("Agent: {} initialized, Device : {}".format(self.name, Agent.device))

    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.get_eps()
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=Agent.device, dtype=torch.long)
    
    def select_max_action(self,state):
        with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        
    def get_eps(self):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        return eps_threshold
    def plot_durations(self,show_result=False):
        return
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title(f'Result {self.name}, Gamma: {Agent.GAMMA}, TAU: {Agent.TAU}, LR: {Agent.LR}, BATCH_SIZE: {Agent.BATCH_SIZE}')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(),block=False)

        plt.pause(0.001)  # pause a bit so that plots are updated
        if Agent.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


    def plot_performance(self,episode_rewards,is_evaluation=False):
        x_axis = np.arange(0,len(episode_rewards))
        plt.plot(x_axis,np.array(episode_rewards),color="r",linestyle="-",marker="o",markersize=1)
        title = f"{self.name} {Agent.NAME_SUFFIX}"
        path = f"results/{self.env.spec.id}/{self.name}_{Agent.NAME_SUFFIX}"
        if(is_evaluation):
            title += "_eval"
            path += "_eval"
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        
        plt.savefig(path)
        #plt.show(block=False)
        