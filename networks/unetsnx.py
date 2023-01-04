import torch
import torch.nn as nn
import math
import warnings
from typing import Any


from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init, trunc_normal_init)

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetrPrUpBlock, UnetOutBlock


class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN3d', requires_grad=True)):
        super(StemConv, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2,
                      kernel_size=3, stride=2, padding=1),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels,
                      kernel_size=3, stride=2, padding=1),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W, D = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W, D


class AttentionModule(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv3d(dim, dim, (1, 7, 7), padding=(0, 3, 3), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (7, 7, 1), padding=(3, 3, 0), groups=dim)

        self.conv1_1 = nn.Conv3d(dim, dim, (1, 11, 11), padding=(0, 5, 5), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (11, 11, 1), padding=(5, 5, 0), groups=dim)

        self.conv2_1 = nn.Conv3d(
            dim, dim, (1, 21, 21), padding=(0, 10, 10), groups=dim)
        self.conv2_2 = nn.Conv3d(
            dim, dim, (21, 21, 1), padding=(10, 10, 0), groups=dim)
        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(BaseModule):

    def __init__(self,
        dim,
        mlp_ratio=4.,
        drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_cfg=dict(type='BN3d', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W, D)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=dict(type='BN3d', requires_grad=True)):
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size)

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W, D = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W, D


class MSCAN(BaseModule):
    def __init__(self,
        in_chans=1,
        embed_dims=[96, 192, 384, 768],
        mlp_ratios=[4, 4, 4, 4],
        drop_rate=0.,
        drop_path_rate=0.,
        depths=[3, 4, 6, 3],
        num_stages=4,
        norm_cfg=dict(type='BN3d', requires_grad=True),
        pretrained=None,
        init_cfg=None):
        super(MSCAN, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(1, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_cfg=norm_cfg)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv3d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:

            super(MSCAN, self).init_weights()

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W, D = patch_embed(x)
            for blk in block:
                x = blk(x, H, W, D)
            x = norm(x)
            x = x.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
            outs.append(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x



class UNETSNX(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        feature_size=48,
        patch_size=4,
        spatial_dims=3,
        norm_name='instance',
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.encoder = MSCAN()

        self.encoder5 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder0 = nn.ConvTranspose3d(
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x):
        enc1, enc2, enc3, enc4 = self.encoder(x)
        dec4 = self.encoder5(enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        dec0 = self.decoder0(dec1)
        return self.out(dec0)