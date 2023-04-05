import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.stochastic_depth import StochasticDepth

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

from .blocks.utils import LayerNorm
from .blocks.inceptionnext import MetaNeXtBlock, InceptionDWConv3d, MlpHead
from timm.models.layers import trunc_normal_


class UNETMNX_A2(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            feature_size=24,
            patch_size=4,
            spatial_dims=3,
            norm_name='instance',
            stochastic_depth_prob=0.4,
            depths=[3, 3, 9, 3],
            **kwargs
    ) -> None:
        super().__init__()
        print(depths)

        self.backbone = MetaNeXt(
            depths=(3, 3, 9, 3), 
            dims=(96, 192, 384, 768)
        )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.inc = self.backbone.stem

        self.encoder1 = self.backbone.stages[0]
        self.encoder2 = self.backbone.stages[1]
        self.encoder3 = self.backbone.stages[2]
        self.encoder4 = self.backbone.stages[3]

        self.encoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 32,
            out_channels=feature_size * 16,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=patch_size,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x):
        enc0 = self.encoder0(x)

        inc = self.inc(x)

        enc1 = self.encoder1(inc)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = self.encoder5(enc4, enc3)

        dec3 = self.decoder4(dec4, enc2)
        dec2 = self.decoder3(dec3, enc1)
        dec1 = self.decoder2(dec2, inc)
        dec0 = self.decoder1(dec1, enc0)
        out = self.out(dec0)
        return out

    def encode(self, x):
        inc = self.inc(x)
        enc1 = self.encoder1(inc)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        return enc4


class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs, data_format="channels_first"),
                nn.Conv3d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=in_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.blocks(x)
        x = self.downsample(x)
        return x


class MetaNeXt(nn.Module):
    r""" MetaNeXt
        A PyTorch impl of : `InceptionNeXt: When Inception Meets ConvNeXt`  - https://arxiv.org/pdf/2203.xxxxx.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 9, 3)
        dims (tuple(int)): Feature dimension at each stage. Default: (96, 192, 384, 768)
        token_mixers: Token mixer function. Default: nn.Identity
        norm_layer: Normalziation layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        head_fn: classifier head
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            in_chans=1,
            num_classes=1000,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            token_mixers=InceptionDWConv3d,
            norm_layer=LayerNorm,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            head_fn=MlpHead,
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            **kwargs,
    ):
        super().__init__()

        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage


        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0] // 2, kernel_size=4, stride=4),
            norm_layer(dims[0] // 2, data_format="channels_first"),
        )

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0] // 2
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]
            stages.append(MetaNeXtStage(
                prev_chs,
                out_chs,
                ds_stride=2, 
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
            ))
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.apply(self._init_weights)
    
    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

