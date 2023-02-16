import torch
import torch.nn as nn
from torch.nn import InstanceNorm3d

from torchvision.ops.stochastic_depth import StochasticDepth

from timm.models.layers import DropPath

from .utils import LayerNorm, Permute, GRN


class DiConvNeXt(nn.Module):
    def __init__(
        self,
        dim=48,
        layer_scale=1e-6,
        stochastic_depth_prob=0.0,
        dilation=1,
        norm_name='layer'
    ) -> None:
        super().__init__()
        if norm_name == 'instance':
            norm = InstanceNorm3d(dim, eps=1e-6)
        elif norm_name == 'layer':
            norm = LayerNorm(dim, eps=1e-6)
        else:
            raise ValueError(f'invalid norm_name: {norm_name} !')

        kernel_size = 7 if dilation == 1 else 3
        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=3, groups=dim, bias=True, dilation=dilation),
            Permute([0, 2, 3, 4, 1]),
            norm,
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
    """ ConvNeXtV2 Block.
    """

    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        dilation=1,
        norm_name='layer'
    ):
        super().__init__()
        
        if norm_name == 'instance':
            norm = InstanceNorm3d(dim, eps=1e-6)
        elif norm_name == 'layer':
            norm = LayerNorm(dim, eps=1e-6)
        else:
            raise ValueError(f'invalid norm_name: {norm_name} !')

        kernel_size = 7 if dilation == 1 else 3
        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=3, groups=dim, bias=True, dilation=dilation),
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

    
class LargeDiConvNeXtV2(nn.Module):
    """ ConvNeXtV2 Block.
    """

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
