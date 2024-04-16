
from layers.vbl_base import VBLinear
import torch
import torch.nn.functional as F

class AdinVBL(VBLinear):
    name= "Adin Variational Bayesian Linear Layers"
    def __init__(self,in_features : int,out_features : int,bias : bool = True,prior_log_sig2 = 0.4):
        super(AdinVBL, self).__init__(in_features=in_features,out_features=out_features,bias=bias,prior_log_sig2=prior_log_sig2)
    #uses local reparametrization trick to reduce variance

    def forward(self, input):
        output_mu = F.linear(input, self.weight_mu, self.bias_mu)
        output_sig2 = F.linear(input.pow(2), self.weight_log_sig2.exp(), self.bias_log_sig2.clamp(-11,11).exp())

        return output_mu + output_sig2.sqrt() * torch.randn_like(output_sig2) #,output_mu,output_sig2
