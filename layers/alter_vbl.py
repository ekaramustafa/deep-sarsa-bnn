from layers.vbl_base import VBLinear
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlterVBL(VBLinear):
    name= "Alter Variational Bayesian Linear"
    def __init__(self,in_features: int, out_features: int, bias: bool = True, prior_log_sig2 =0.4):
        super(AlterVBL, self).__init__(in_features=in_features,out_features=out_features,bias=bias,prior_log_sig2=prior_log_sig2)

    #directly samples from the parameters without local reparametrization trick
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