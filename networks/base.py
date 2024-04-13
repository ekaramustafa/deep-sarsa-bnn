import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions,n_hidden=128):
        super(QNetwork, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_hidden = n_hidden

    def forward(self, x):
        print("BaseQNetwork forward method called, please implement this method in the derived class")
        return