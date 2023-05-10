import torch
import torch.nn as nn
from torch.nn import InstanceNorm3d

from torchvision.ops.stochastic_depth import StochasticDepth

from timm.models.layers import DropPath

import numpy as np
from monai.networks.blocks import UnetResBlock
from monai.networks.nets.swin_unetr import SwinTransformerBlock, get_window_size, compute_mask
from monai.utils import ensure_tuple_rep, optional_import
rearrange, _ = optional_import("einops", name="rearrange")

from .convnext import DiConvNeXt
from .conv2former import Conv2FormerBlock, ConvMod
from .utils import LayerNorm, Permute


class ConvSwinTransformerBlock_A0(nn.Module):
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
    
    
class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features=48,
        out_features=None,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6
    ):
        super().__init__()
        out_features = out_features if out_features else in_features
        
        self.block = nn.Sequential(
            Permute([0, 2, 3, 4, 1]),
            LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features=in_features, out_features=4 * out_features, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * out_features, out_features=out_features, bias=True),
            Permute([0, 4, 1, 2, 3]),
        )
        self.layer_scale = nn.Parameter(torch.ones(out_features, 1, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        return result


class WideFocusModule(nn.Module): 
    """
    Wide-Focus module.
    https://arxiv.org/ftp/arxiv/papers/2206/2206.00566.pdf
    https://github.com/kingo233/FCT-Pytorch/blob/main/utils/model.py
    """
    def __init__(self, dim, drop_rate=0.1):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        self.dil_conv2 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding="same", dilation=2),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        self.dil_conv3 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding="same", dilation=3),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.dil_conv2(x)
        x3 = self.dil_conv3(x)
        added = x1 + x2 + x3
        x_out = self.conv4(added)
        return x_out
    

class WideFocusBlock(nn.Module):
    def __init__(self, dim, drop_rate=0.1, drop_path=0.0):
        super().__init__()
        self.wide_forcus = WideFocusModule(dim, drop_rate)
        self.norm = LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        # layer nrom
        x2 = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x2 = self.norm(x2)
        x2 = x2.permute(0, 4, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)
        # wide forcus
        x3 = self.wide_forcus(x2)
        out = x + x3
        return out

class ConvAttnWideFocusBlock(nn.Module):
    def __init__(self, dim, drop_rate=0.1, drop_path=0.0):
        super().__init__()
        self.conv_attn = ConvMod(dim)
        self.wide_forcus = WideFocusModule(dim, drop_rate)
        
        self.norm = LayerNorm(dim, eps=1e-6)
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1, 1) * 1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # conv attn
        x1 = x + self.drop_path(self.layer_scale * self.conv_attn(x))
        # layer nrom
        x2 = x1.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x2 = self.norm(x2)
        x2 = x2.permute(0, 4, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)
        # wide forcus
        x3 = self.wide_forcus(x2)
        out = x1 + x3
        return out


class ConvBlock_A1(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1
    ):
        super().__init__()
        self.conv2former_block = Conv2FormerBlock(dim, drop_path=stochastic_depth_prob)
        self.convnext_block = DiConvNeXt(dim, stochastic_depth_prob, kernel_size, dilation)
    
    def forward(self, x):
        x1 = self.conv2former_block(x)
        x2 = self.convnext_block(x)
        
        y = x + x1 + x2
        
        return y
    
    
class ConvBlock_A2(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1
    ):
        super().__init__()
        self.conv2former_block = Conv2FormerBlock(dim, drop_path=stochastic_depth_prob)
        self.convnext_block = DiConvNeXt(dim, stochastic_depth_prob, kernel_size, dilation)
        self.mlp_block = MLPBlock(dim*2, dim, stochastic_depth_prob=stochastic_depth_prob)
    
    def forward(self, x):
        x1 = self.conv2former_block(x)
        x2 = self.convnext_block(x)
        
        y = x + self.mlp_block(torch.cat((x1, x2), dim=1))
        return y
    
    
class ConvBlock_A3(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1
    ):
        super().__init__()
        self.conv2former_block = Conv2FormerBlock(dim, drop_path=stochastic_depth_prob)
        self.convnext_block = DiConvNeXt(dim, stochastic_depth_prob, kernel_size, dilation)
        self.mlp_block = MLPBlock(dim, stochastic_depth_prob=stochastic_depth_prob)
    
    def forward(self, x):
        x1 = self.conv2former_block(x)
        x2 = self.convnext_block(x)
        
        y = x + self.mlp_block(x1 + x2)
        return y
    
    
class ConvBlock_A4(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1
    ):
        super().__init__()
        self.conv2former_block = Conv2FormerBlock(dim, drop_path=stochastic_depth_prob)
        self.convnext_block = DiConvNeXt(dim, stochastic_depth_prob, kernel_size, dilation)
        self.res_block = UnetResBlock(
            spatial_dims=3, 
            in_channels=dim*2, 
            out_channels=dim, 
            kernel_size=3, 
            stride=1, 
            norm_name="instance"
        )
    
    def forward(self, x):
        x1 = self.conv2former_block(x)
        x2 = self.convnext_block(x)
        
        y = x + self.res_block(torch.cat((x1, x2), dim=1))
        return y 
    
    
class ConvBlock_A5(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1
    ):
        super().__init__()
        self.conv2former_block = Conv2FormerBlock(dim, drop_path=stochastic_depth_prob)
        self.convnext_block = DiConvNeXt(dim, stochastic_depth_prob, kernel_size, dilation)
        self.conv_block = nn.Conv3d(dim*2, dim, 1)
        
        
    
    def forward(self, x):
        x1 = self.conv2former_block(x)
        x2 = self.convnext_block(x)
        
        y = x + self.conv_block(torch.cat((x1, x2), dim=1))
        return y
    

class ConvBlock_A6(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1
    ):
        super().__init__()
        self.conv2former_block = Conv2FormerBlock(dim, drop_path=stochastic_depth_prob)
        self.convnext_block = DiConvNeXt(dim, stochastic_depth_prob, kernel_size, dilation)
    
    def forward(self, x):
        x = self.conv2former_block(x)
        y = self.convnext_block(x)
        
        return y


class ConvBlock_A7(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1
    ):
        super().__init__()
        self.conv2former_block = Conv2FormerBlock(dim, drop_path=stochastic_depth_prob)
        self.convnext_block = DiConvNeXt(dim, stochastic_depth_prob, kernel_size, dilation)
    
    def forward(self, x):
        x = self.convnext_block(x)
        y = self.conv2former_block(x)
        
        return y    
    
    
class ConvBlock_A8(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=7,
        dilation=1
    ):
        super().__init__()
        self.convnext_block = DiConvNeXt(dim, stochastic_depth_prob, kernel_size, dilation)
        self.wf_block = WideFocusBlock(dim)
    
    def forward(self, x):
        x1 = self.convnext_block(x)
        y = x1 + self.wf_block(x1)
        return y
    

class ConvBlock_A9(nn.Module):
    def __init__(
        self,
        dim=48,
        stochastic_depth_prob=0.0,
        kernel_size=[7, 3],
        dilation=[1, 3]
    ):
        super().__init__()
        
        blocks = [DiConvNeXt(dim, stochastic_depth_prob, k, d) for k, d in zip(kernel_size, dilation)]
        self.blocks  = nn.Sequential(*blocks)
        
    
    def forward(self, x):
        y = self.blocks(x)
        return y


class ConvNeXtBlock_V1(nn.Module):
    def __init__(
        self,
        dim=48,
        kernel_size=7,
        dilation=1,
        exp_rate=4,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6,
        norm_name='layer'
    ):
        super().__init__()
        
        padding = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        
        if norm_name == 'layer':
            norm = LayerNorm(dim, eps=1e-6)
        elif norm_name == 'group':
            num_group = 4
            norm = nn.GroupNorm(num_group, dim)
        else:
            raise ValueError(f'invalid norm name: {norm_name}')
            

        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=True, dilation=dilation),
            Permute([0, 2, 3, 4, 1]),
            norm,
            nn.Linear(in_features=dim, out_features=exp_rate * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=exp_rate * dim, out_features=dim, bias=True),
            Permute([0, 4, 1, 2, 3]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result