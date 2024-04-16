import torch.nn as nn
import torch.functional as F
from networks.base import QNetwork  

# Deep Q Network with Lineary Layers
class DQNLinear(QNetwork):

    def __init__(self, n_observations, n_actions,n_hidden=128):
        super(DQNLinear, self).__init__(n_actions=n_actions)
        self.layer1 = nn.Linear(n_observations, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

"""
The input to the neural
network consists is an 84 x 84 x 4 image produced by φ. The first hidden layer convolves 16 8 x 8
filters with stride 4 with the input image and applies a rectifier nonlinearity [10, 18]. The second
hidden layer convolves 32 4 x 4 filters with stride 2, again followed by a rectifier nonlinearity. 
The final hidden layer is fully-connected and consists of 256 rectifier units. 
The output layer is a fully-connected linear layer with a single output for each valid action. 
The number of valid actions varied between 4 and 18 on the games we considered.

Playing Atari with Deep Reinforcement Learning
ref: https://arxiv.org/pdf/1312.5602.pdf
"""

class DQNConv2d(QNetwork):

    def __init__(self, stack_dim, n_actions):
        super(DQNConv2d, self).__init__(n_actions=n_actions)
        self.network = nn.Sequential(
            #cnn
            nn.Conv2d(stack_dim, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            #classifier part
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)
    

