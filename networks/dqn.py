import torch.nn as nn
import torch.functional as F
from base import QNetwork  

class DQN(QNetwork):

    def __init__(self, n_observations, n_actions,n_hidden=128):
        super(DQN, self).__init__(n_observations=n_observations, n_actions=n_actions,n_hidden=n_hidden)
        self.layer1 = nn.Linear(n_observations, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)