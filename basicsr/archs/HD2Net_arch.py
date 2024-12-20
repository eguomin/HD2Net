import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from .RCAN3D_arch import RCAN3D

@ARCH_REGISTRY.register()
class HD2Net(nn.Module):
    def __init__(self, input_channel, num_channels, num_blocks, num_groups, reduction, residual_scaling,
                 num_output_channels,return_midlq):
        super(HD2Net, self).__init__()
        self.return_midlq = return_midlq
        self.denoiseBlock = RCAN3D(input_channel, num_channels, num_blocks, 3, reduction,residual_scaling,
                                 num_output_channels)
        self.deabeBlock = RCAN3D(input_channel, num_channels, num_blocks, 5, reduction, residual_scaling,
                                 num_output_channels)
    def forward(self, x):
        self.midlq = self.denoiseBlock(x)
        self.out = self.deabeBlock(self.midlq)
        if self.return_midlq:
            return self.out,self.midlq
        else:
            return self.out

