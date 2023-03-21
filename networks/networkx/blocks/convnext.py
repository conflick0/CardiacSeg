import torch
import torch.nn as nn
from torch.nn import InstanceNorm3d

from torchvision.ops.stochastic_depth import StochasticDepth

from timm.models.layers import DropPath


from .cbam import CBAM
from .utils import LayerNorm, Permute, GRN
    

class DiConvNeXt(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1,
        layer_scale=1e-6
    ):
        super().__init__()
        
        padding = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2

        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=True, dilation=dilation),
            Permute([0, 2, 3, 4, 1]),
            LayerNorm(dim, eps=1e-6),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 4, 1, 2, 3]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result

    
class DiConvNeXtV2(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1
    ):
        super().__init__()
        
        padding = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2

        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=True, dilation=dilation),
            Permute([0, 2, 3, 4, 1]),
            LayerNorm(dim, eps=1e-6),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            GRN(4 * dim),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 4, 1, 2, 3]),
        )

        self.stochastic_depth = DropPath(stochastic_depth_prob) if stochastic_depth_prob > 0. else nn.Identity()

    def forward(self, input):
        result = self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class ResDiConvNeXt(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_probs=[0.0, 0.0],
        kernel_sizes=[7, 3],
        dilations=[1, 3]
    ):
        super().__init__()
        
        block = DiConvNeXt
        
        blocks = []
        for s, k, d in zip(stochastic_depth_probs, kernel_sizes, dilations):
            blocks.append(block(dim, s, k, d))
            
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        result = self.blocks[0](input)
        result = self.blocks[1](result)
        result += input
        
        return result
    

class DiConvNeXtSkipCBAM(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_probs=[0.0, 0.0],
        kernel_sizes=[7, 3],
        dilations=[1, 3]
    ):
        super().__init__()
        
        block = DiConvNeXt
        self.cbam = CBAM(dim, reduction=16, kernel_size=7)
        
        blocks = []
        for s, k, d in zip(stochastic_depth_probs, kernel_sizes, dilations):
            blocks.append(block(dim, s, k, d))
            
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        result = self.blocks[0](input)
        result = self.blocks[1](result)
        result = input + self.cbam(result)
        
        return result
    
    
class DiConvNeXtCBAM(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_probs=[0.0, 0.0],
        kernel_sizes=[7, 3],
        dilations=[1, 3]
    ):
        super().__init__()
        
        block = DiConvNeXt
        self.cbam = CBAM(dim, reduction=16, kernel_size=7)
        
        blocks = []
        for s, k, d in zip(stochastic_depth_probs, kernel_sizes, dilations):
            blocks.append(block(dim, s, k, d))
            
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        result = self.blocks[0](input)
        result = self.blocks[1](result)
        result = self.cbam(result)
        
        return result
    

class DDiConvNeXt(nn.Module):
    '''double Di ConvNeXt'''
    def __init__(
        self,
        dim=48,
        stochastic_depth_probs=[0.0, 0.0],
        kernel_sizes=[7, 3],
        dilations=[1, 3]
    ):
        super().__init__()
        
        block = DiConvNeXt
        
        blocks = []
        for s, k, d in zip(stochastic_depth_probs, kernel_sizes, dilations):
            blocks.append(block(dim, s, k, d))
            
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        result = self.blocks[0](input)
        result = self.blocks[1](result)
        
        return result
