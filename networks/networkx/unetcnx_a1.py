import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.stochastic_depth import StochasticDepth

from timm.models.layers import trunc_normal_

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

from .blocks.convnext_v2 import ConvNeXtBlock_V2
from .blocks.utils import LayerNorm


class UNETCNX_A1(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            patch_size=4,
            kernel_size=7,
            exp_rate=4,
            feature_size=48,
            depths=[3, 3, 9, 3],
            drop_path_rate=0.0, 
            use_init_weights=False,
            is_conv_stem=False,
            **kwargs
    ) -> None:
        super().__init__()
        
        feature_sizes = [feature_size*(2**i) for i in range(len(depths))]
        
        first_feature_size = feature_sizes[0]
        
        decoder_norm_name = 'instance' 
        res_block = True
        spatial_dims = 3
        
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=first_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        
        self.backbone = Backbone(
            in_channels=in_channels,
            patch_size=patch_size,
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            feature_sizes=feature_sizes,
            depths=depths,
            drop_path_rate=drop_path_rate,
            use_init_weights=use_init_weights,
            is_conv_stem=is_conv_stem
        )
        
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3],
            out_channels=feature_sizes[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[2],
            out_channels=feature_sizes[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[1],
            out_channels=feature_sizes[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[0],
            out_channels=first_feature_size,
            kernel_size=3,
            upsample_kernel_size=patch_size,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )


        self.out_block = UnetOutBlock(spatial_dims=3, in_channels=first_feature_size, out_channels=out_channels)

    def forward(self, x):
        enc0 = self.encoder0(x)
        
        hidden_states_out = self.backbone(x)
        
        enc1 = hidden_states_out[0]
        enc2 = hidden_states_out[1]
        enc3 = hidden_states_out[2]
        enc4 = hidden_states_out[3]
        
        # print('e0:', enc0.shape)
        # print('e1:', enc1.shape)
        # print('e2:', enc2.shape)
        # print('e3:', enc3.shape)
        # print('e4:', enc4.shape)
        
        dec4 = self.decoder4(enc4, enc3)
        dec3 = self.decoder3(dec4, enc2)
        dec2 = self.decoder2(dec3, enc1)
        dec1 = self.decoder1(dec2, enc0)
        
        # print('d4:', dec4.shape)
        # print('d3:', dec3.shape)
        # print('d2:', dec2.shape)
        # print('d1:', dec1.shape)
        
        out = self.out_block(dec1)
        return out

    def encode(self, x):
        hidden_states_out = self.backbone(x)
        return hidden_states_out[3]


class Backbone(nn.Module):
    def __init__(
        self,
        in_channels=1,
        patch_size=4,
        kernel_size=7,
        exp_rate=4,
        feature_sizes=[48, 96, 192, 384],
        depths=[2, 2, 2, 2],
        drop_path_rate=0.0,
        use_init_weights=False,
        is_conv_stem=False
    ):
        super().__init__()
        
        print('patch size:', patch_size)
        print('ker size:', kernel_size)
        print('exp rate:', exp_rate)
        print('feature sizes:', feature_sizes)
        print('depths:', depths)
        print('drop rate:',drop_path_rate)
        print('use init weights:', use_init_weights)
        print('is conv stem:', is_conv_stem)
        
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        
        if is_conv_stem:
            stem = nn.Sequential(
                nn.Conv3d(in_channels, feature_sizes[0], kernel_size=7, stride=patch_size, padding=3),
                LayerNorm(feature_sizes[0], eps=1e-6, data_format="channels_first")
            )
        else:
             stem = nn.Sequential(
                nn.Conv3d(in_channels, feature_sizes[0], kernel_size=patch_size, stride=patch_size),
                LayerNorm(feature_sizes[0], eps=1e-6, data_format="channels_first")
            )
        
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(feature_sizes[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(feature_sizes[i], feature_sizes[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock_V2(
                        dim=feature_sizes[i], 
                        kernel_size=kernel_size,
                        exp_rate=exp_rate,
                        drop_path=dp_rates[cur + j],
                    )
                for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        if use_init_weights:
            # init weight
            print('use init weights')
            self.apply(self._init_weights)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        return outs
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

