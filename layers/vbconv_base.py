import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

#Variational Bayesian Convoluational Layer
class VBConv2d(nn.Module):
    __constants__ = ['in_channels', 'out_channels', 'kernel_size']
    in_channels: int
    out_channels: int
    kernel_size: _pair
    weight: torch.Tensor
    name = "Variational Bayesian 2D Convolutional Layers"

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _pair, stride: _pair = 1,
                 padding: _pair = 0, dilation: _pair = 1, groups: int = 1, bias: bool = True,
                 prior_log_sig2=0.4) -> None:
        super(VBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        self.weight_mu = nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)))
        self.weight_log_sig2 = nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)))
        self.weight_mu_prior = nn.Parameter(torch.zeros((out_channels, in_channels, *kernel_size)), requires_grad=False)
        self.weight_log_sig2_prior = nn.Parameter(prior_log_sig2 * torch.zeros((out_channels, in_channels, *kernel_size)),
                                                  requires_grad=False)
        if self.has_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_log_sig2 = nn.Parameter(torch.Tensor(out_channels))
            self.bias_mu_prior = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            self.bias_log_sig2_prior = nn.Parameter(prior_log_sig2 * torch.zeros(out_channels), requires_grad=False)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sig2', None)
        self.reset_parameters()

    def reset_parameters(self,) -> None:
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5) / math.sqrt(n))
        init.constant_(self.weight_log_sig2, -10)
        if self.has_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_mu, -bound, bound)
            init.constant_(self.bias_log_sig2, -10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    def get_mean_var(self, input: torch.Tensor) -> torch.Tensor:
        mu = F.conv2d(input, self.weight_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
        sig2 = F.conv2d(input.pow(2), self.weight_log_sig2.exp(), self.bias_log_sig2.exp(), self.stride, self.padding,
                        self.dilation, self.groups)
        return mu, sig2

    def extra_repr(self) -> str:
        return 'in_channels={}, out_channels={}, kernel_size={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.has_bias
        )

    def update_prior(self, newprior):
        self.weight_mu_prior.data = newprior.weight_mu.data.clone()
        self.weight_mu_prior.data.requires_grad = False
        self.weight_log_sig2_prior.data = newprior.weight_log_sig2.data.clone()
        self.weight_log_sig2_prior.data.requires_grad = False
        if self.has_bias:
            self.bias_mu_prior.data = newprior.bias_mu.data.clone()
            self.bias_mu_prior.data.requires_grad = False
            self.bias_log_sig2_prior.data = newprior.bias_log_sig2.data.clone()
            self.bias_log_sig2_prior.data.requires_grad = False

    def kl_loss(self):
        kl_weight = 0.5 * (
                    self.weight_log_sig2_prior - self.weight_log_sig2 + (
                        self.weight_log_sig2.exp() + (self.weight_mu_prior - self.weight_mu) ** 2) / self.weight_log_sig2_prior.exp() - 1.0)
        kl = kl_weight.sum()
        n = len(self.weight_mu.view(-1))
        if self.has_bias:
            kl_bias = 0.5 * (
                        self.bias_log_sig2_prior - self.bias_log_sig2 + (
                            self.bias_log_sig2.exp() + (self.bias_mu_prior - self.bias_mu) ** 2) / (
                                self.bias_log_sig2_prior.exp()) - 1.0)
            kl += kl_bias.sum()
            n += len(self.bias_mu.view(-1))
        return kl, n
