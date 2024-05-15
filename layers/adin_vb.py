import torch
import torch.nn.functional as F
#From AdinLab, I don't know how to cite a private repository

# Adin Variational Bayesian Linear Layer 
from layers.vbl_base import VBLinear

class AdinVBL(VBLinear):
    name= "Adin Variational Bayesian Linear Layers"
    def __init__(self,in_features : int,out_features : int,bias : bool = True,prior_log_sig2 = 0.4):
        super(AdinVBL, self).__init__(in_features=in_features,out_features=out_features,bias=bias,prior_log_sig2=prior_log_sig2)
    #uses local reparametrization trick to reduce variance

    def forward(self, input):
        output_mu = F.linear(input, self.weight_mu, self.bias_mu)
        output_sig2 = F.linear(input.pow(2), self.weight_log_sig2.exp(), self.bias_log_sig2.clamp(-11,11).exp())

        return output_mu + output_sig2.sqrt() * torch.randn_like(output_sig2) #,output_mu,output_sig2



# Adin Variational Bayesian 2D Convolutional Layer
from layers.vbconv_base import VBConv2d

class AdinVBConv2d(VBConv2d):
    name = "Adin Variational Bayesian 2D Convolutional Layers"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, prior_log_sig2=0.4):
        
        super(AdinVBConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, 
                                           padding, dilation, groups, bias, prior_log_sig2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu, sig2 = self.get_mean_var(input)
        eps = torch.randn_like(mu)
        return mu + torch.sqrt(sig2) * eps

