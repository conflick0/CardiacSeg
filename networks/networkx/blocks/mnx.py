import torch
import torch.nn as nn
from torch.nn import InstanceNorm3d

from torchvision.ops.stochastic_depth import StochasticDepth


import numpy as np


class MedNeXtBlock(nn.Module):
    def __init__(
        self,
        dim=48,
        kernel_size=7,
        dilation=1,
        exp_rate=4,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6
    ):
        super().__init__()
        
        padding = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2

        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, dilation=dilation),
            nn.GroupNorm(dim, dim),
            nn.Conv3d(dim, dim * exp_rate, kernel_size=1, groups=dim),
            nn.GELU(),
            nn.Conv3d(dim* exp_rate, dim, kernel_size=1, groups=dim),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result

    
class MedNeXtCatBlock(nn.Module):
    def __init__(
        self,
        dim=48,
        kernel_size=7,
        dilation=1,
        exp_rate=4,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6
    ):
        super().__init__()
        
        self.dim =dim*2
        
        padding = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2

        self.block = nn.Sequential(
            nn.Conv3d(dim*2, dim, kernel_size=kernel_size, padding=padding, groups=dim, dilation=dilation),
            nn.GroupNorm(dim, dim),
            nn.Conv3d(dim, dim * exp_rate, kernel_size=1, groups=dim),
            nn.GELU(),
            nn.Conv3d(dim* exp_rate, dim, kernel_size=1, groups=dim),
        )
        
        self.skip_conv = nn.Conv3d(dim*2, dim, kernel_size=1, groups=dim)
        
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input, skip_input):
        result = torch.cat([input, skip_input], dim=1)
        skip_result = self.skip_conv(result)
        
        result = self.layer_scale * self.block(result)
        result = self.stochastic_depth(result)
        result += skip_result
        
        return result
    
    
class MedNeXtDownBlock(nn.Module):
    def __init__(
        self,
        dim=48,
        kernel_size=7,
        exp_rate=4,
        stride=2,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6
    ):
        super().__init__()
        
        padding = (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, stride=stride),
            nn.GroupNorm(dim, dim),
            nn.Conv3d(dim, dim * exp_rate, kernel_size=1, groups=dim),
            nn.GELU(),
            nn.Conv3d(dim* exp_rate, dim * 2, kernel_size=1, groups=dim),
        )
        
        self.skip_conv = nn.Conv3d(dim,  dim * 2, kernel_size=1, groups=dim, stride=stride)
        
        self.layer_scale = nn.Parameter(torch.ones(dim * 2, 1, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        skip_result = self.skip_conv(input)
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += skip_result
        return result
    
    
    
class MedNeXtUpBlock(nn.Module):
    def __init__(
        self,
        dim=48,
        kernel_size=7,
        exp_rate=4,
        stride=2,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6
    ):
        super().__init__()
        
        padding = (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.ConvTranspose3d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, stride=stride, output_padding=kernel_size % 2),
            nn.GroupNorm(dim, dim),
            nn.Conv3d(dim, dim * exp_rate, kernel_size=1, groups=dim),
            nn.GELU(),
            nn.Conv3d(dim* exp_rate, dim // 2, kernel_size=1, groups=dim // 2),
        )
        
        self.skip_conv = nn.ConvTranspose3d(dim, dim // 2, kernel_size=1, groups=dim // 2, stride=stride, padding=1//2, output_padding=1 % 2)
        
        self.layer_scale = nn.Parameter(torch.ones(dim // 2, 1, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        skip_result = self.skip_conv(input)
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += skip_result
        return result

    

class MedNeXtUpBlocks(nn.Module):
    def __init__(
        self,
        dim=48,
        kernel_size=7,
        dilation=1,
        exp_rate=4,
        stride=2,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6,
        depths=2,
    ):
        super().__init__()
        # up block
        self.up_block = MedNeXtUpBlock(
            dim=dim*2,
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            stride=stride,
            stochastic_depth_prob=stochastic_depth_prob,
            layer_scale=layer_scale
        )
        
        # add cat block
        self.cat_block = MedNeXtCatBlock(
            dim=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            exp_rate=exp_rate,
            stochastic_depth_prob=stochastic_depth_prob,
            layer_scale=layer_scale
        )
        
        blocks = []
        for i in range(depths-1):
            blocks.append(MedNeXtBlock(
                dim=dim,
                kernel_size=kernel_size,
                dilation=dilation,
                exp_rate=exp_rate,
                stochastic_depth_prob=stochastic_depth_prob,
                layer_scale=layer_scale
            ))
        
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input, skip_input):
        result = self.up_block(input)
        result = self.cat_block(result, skip_input)
        result = self.blocks(result)
        return result

