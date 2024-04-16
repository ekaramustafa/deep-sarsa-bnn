#Base class for Bayesian Neural Network
import torch.nn as nn
import torch.nn.functional as F
from networks.base import QNetwork 

#Bayesian Neural Networks (Stochastic Q-Networks)

# Deep Q Network with with Variational Bayesian Linear Layers
"""
This layer can be used for both in linear and convolutional neural networks.
Therefore, the output_dim and n_actions should be modified and used accordingly.
This setup is for using linear layers in CNNs. 
"""
class DQN_VBLinear(QNetwork):
    name = "DQN with Variational Bayesian Linear Layers"
    
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


"""
The input to the neural network consists is an 84 x 84 x 4 image produced by φ. 
The first hidden layer convolves 16 8 x 8 filters with stride 4 with the input image 
and applies a rectifier nonlinearity [10, 18]. The second hidden layer 
convolves 32 4 x 4 filters with stride 2, again followed by a rectifier nonlinearity. 
The final hidden layer is fully-connected and consists of 256 rectifier units. 
The output layer is a fully-connected linear layer with a single output for each valid action. 
The number of valid actions varied between 4 and 18 on the games we considered.

Playing Atari with Deep Reinforcement Learning
ref: https://arxiv.org/pdf/1312.5602.pdf
"""
# Deep Q Network with with Variational Bayesian Linear Layers
class DQN_VBConv2D(QNetwork):
    name = "DQN with Convolutional Layers"
    def __init__(self, n_actions, conv_layer_class, linear_layer_class):
       super().__init__(n_actions=n_actions)
       DQN_VBConv2D.name = f"DQN with {conv_layer_class.name} and {linear_layer_class.name} Layers"
       self.network = nn.Sequential(
            #cnn
            conv_layer_class(4, 32, kernel_size=(8,8), stride=4),
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
    
