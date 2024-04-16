import torch

from vbconv_base import VBConv2d

class AdinVBConv2d(VBConv2d):
    name = "Adin Variational Bayesian 2D Convolutional Layer"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, prior_log_sig2=0.4):
        
        super(AdinVBConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, 
                                           padding, dilation, groups, bias, prior_log_sig2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu, sig2 = self.get_mean_var(input)
        eps = torch.randn_like(mu)
        return mu + torch.sqrt(sig2) * eps