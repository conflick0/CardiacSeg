import torch
import torch.nn as nn
from torch.nn import InstanceNorm3d

from torchvision.ops.stochastic_depth import StochasticDepth

from timm.models.layers import DropPath

import numpy as np
from monai.networks.nets.swin_unetr import SwinTransformerBlock, get_window_size, compute_mask
from monai.utils import ensure_tuple_rep, optional_import
rearrange, _ = optional_import("einops", name="rearrange")

from .convnext import DiConvNeXt


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
