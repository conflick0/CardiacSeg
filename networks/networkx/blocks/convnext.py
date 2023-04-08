import torch
import torch.nn as nn
from torch.nn import InstanceNorm3d

from torchvision.ops.stochastic_depth import StochasticDepth

from timm.models.layers import DropPath

import numpy as np
from monai.networks.nets.swin_unetr import SwinTransformerBlock, get_window_size, compute_mask
from monai.utils import ensure_tuple_rep, optional_import
rearrange, _ = optional_import("einops", name="rearrange")

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
    
    
class ConvNeXtCBAM(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1
    ):
        super().__init__()
        
        self.block = DiConvNeXt(dim, stochastic_depth_prob, kernel_size, dilation)
        self.cbam = CBAM(dim, reduction=16, kernel_size=7)

    def forward(self, input):
        result = self.block(input)
        result = self.cbam(result)
        
        return result

    
class ConvSwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1,
        num_heads=3
    ):
        super().__init__()
        self.window_size = ensure_tuple_rep(7, 3)
        self.shift_size = tuple(i // 2 for i in self.window_size)
        self.no_shift = tuple(0 for i in self.window_size)

        self.swin_trnasformer_block = nn.Sequential(*[
            SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
            mlp_ratio=0.4,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=True) 
        for i in range(2)])
        
        self.convnext_block = DiConvNeXt(dim, stochastic_depth_prob, kernel_size, dilation)
        
    def forrward_swin_transformer(self, x):
        x_shape = x.size()
        b, c, d, h, w = x_shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        for blk in self.swin_trnasformer_block:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, -1)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x

    def forward(self, input):
        result1 = self.forrward_swin_transformer(input)
        result2 = self.convnext_block(input)
        
        result = result1 + result2
        
        return result