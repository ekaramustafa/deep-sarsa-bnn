#Base class for Bayesian Neural Network
import torch.nn as nn
import torch.nn.functional as F
from networks.base import QNetwork 

class VBLinearNeuralNetwork(QNetwork):
    def __init__(self, layerClass, input_dim, hidden_dim, output_dim, prior_log_sig2=0.4,bias=True):
        super(VBLinearNeuralNetwork, self).__init__(n_actions=output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_name = layerClass.name

        # Define Bayesian linear layers with variational inference
        self.blinear1 = layerClass(input_dim, hidden_dim, prior_log_sig2=prior_log_sig2,bias=bias)
        self.blinear2 = layerClass(hidden_dim, hidden_dim, prior_log_sig2=prior_log_sig2,bias=bias)
        self.blinear3 = layerClass(hidden_dim, output_dim, prior_log_sig2=prior_log_sig2,bias=bias)

    def forward(self, x):
        x = F.relu(self.blinear1(x))
        x = F.relu(self.blinear2(x))
        x = self.blinear3(x)
        return x

    
