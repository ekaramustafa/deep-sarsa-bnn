#Base class for Bayesian Neural Network
import torch.nn as nn
import torch.nn.functional as F
from networks.base import QNetwork 

class DQN_VBLinear(QNetwork):
    def __init__(self, layer_class, input_dim, hidden_dim, output_dim, prior_log_sig2=0.4,bias=True):
        super(DQN_VBLinear, self).__init__(n_actions=output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_name = layer_class.name

        # Define Bayesian linear layers with variational inference
        self.blinear1 = layer_class(input_dim, hidden_dim, prior_log_sig2=prior_log_sig2,bias=bias)
        self.blinear2 = layer_class(hidden_dim, hidden_dim, prior_log_sig2=prior_log_sig2,bias=bias)
        self.blinear3 = layer_class(hidden_dim, output_dim, prior_log_sig2=prior_log_sig2,bias=bias)

    def forward(self, x):
        x = F.relu(self.blinear1(x))
        x = F.relu(self.blinear2(x))
        x = self.blinear3(x)
        return x

class DQN_VBConv2D(QNetwork):
    def __init__(self, n_stack, n_actions, conv_layer_class, linear_layer_class):
       super().__init__(n_actions=n_actions)
       self.network = nn.Sequential(
            #cnn
            conv_layer_class(n_stack, 32, kernel_size=(8,8), stride=4),
            nn.ReLU(),
            conv_layer_class(32, 64, kernel_size=(4,4), stride=2),
            nn.ReLU(),
            conv_layer_class(64, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            #classifier part
            linear_layer_class(7*7*64, 512),
            nn.ReLU(),
            linear_layer_class(512, n_actions),
        )
       
    def forward(self, x):
        return self.network(x / 255.0) 
    
