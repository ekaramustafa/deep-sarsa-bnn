from layers.vbl_base import VBLinear
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlterVBL(VBLinear):
    name= "Alter Variational Bayesian Linear Layers"
    def __init__(self,in_features: int, out_features: int, bias: bool = True, prior_log_sig2 = 0.4):
        super(AlterVBL, self).__init__(in_features=in_features,out_features=out_features,bias=bias,prior_log_sig2=prior_log_sig2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Sample weights and biases
        weight_eps = torch.randn_like(self.weight_log_sig2)
        weight = self.weight_mu + weight_eps * torch.exp(self.weight_log_sig2)

        if self.has_bias:
            bias_eps = torch.randn_like(self.bias_log_sig2)
            bias = self.bias_mu + bias_eps * torch.exp(self.bias_log_sig2)
        else:
            bias = None

        return F.linear(input, weight, bias)
    
from layers.vbconv_base import VBConv2d

class AlterVBConv2d(VBConv2d):
    name= "Alter Variational Bayesian 2D Convolutional Layers"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, prior_log_sig2=0.4):
        super(AlterVBConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, prior_log_sig2)

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + torch.randn_like(self.weight_mu) * self.weight_log_sig2.exp()
            bias = None
            if self.has_bias:
                bias = self.bias_mu + torch.randn_like(self.bias_mu) * self.bias_log_sig2.exp()
    
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)