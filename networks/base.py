import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super(QNetwork, self).__init__()
        self.n_actions = n_actions

    def forward(self, x):
        print("BaseQNetwork forward method called, please implement this method in the derived class")
        return