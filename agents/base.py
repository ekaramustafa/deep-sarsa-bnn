import math
import random
import torch
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
    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env):
        self.env = env
        self.name = "Base Agent"
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        self.policy_net = QNetwork(self.observation_space.n,self.env.action_space.n,n_hidden=128).to(device) 
    
    def init_message(self):
        print("Agent: {} initialized, Device : {}".format(self.name, self.device))

    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([random.randrange(self.env.action_space.n)], device=self.device, dtype=torch.long)
        
    def plot_durations(self,show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
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
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if Agent.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
        