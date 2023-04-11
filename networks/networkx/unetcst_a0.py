import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.stochastic_depth import StochasticDepth

from timm.models.layers import trunc_normal_

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

from .blocks.cst import ConvSwinTransformerBlock_A0
from .blocks.utils import LayerNorm, DilBlockConfig


class UNETCST_A0(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            feature_size=24,
            patch_size=4,
            spatial_dims=3,
            norm_name='instance',
            stochastic_depth_prob=0.1,
            depths=[3, 3, 9, 3],
            num_heads=[3, 6, 12, 24],
            **kwargs
    ) -> None:
        super().__init__()
        print('patch size:', patch_size)
        print('drop rate:',stochastic_depth_prob)
        print('depths:', depths)
        block_setting = [
                DilBlockConfig(feature_size * 2, feature_size * 4, depths[0], [1], [7]),
                DilBlockConfig(feature_size * 4, feature_size * 8, depths[1], [1], [7]),
                DilBlockConfig(feature_size * 8, feature_size * 16, depths[2], [1], [7]),
                DilBlockConfig(feature_size * 16, feature_size * 32, depths[3], [1], [7])
        ]

        self.features = Backbone(
            in_channels=in_channels,
            feature_size=feature_size,
            patch_size=patch_size,
            stochastic_depth_prob=stochastic_depth_prob,
            num_heads=num_heads,
            block_setting=block_setting,
        ).features

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.stem = self.features[0]

        self.encoder1 = nn.Sequential(
            self.features[1],
            self.features[2],
        )

        self.encoder2 = nn.Sequential(
            self.features[3],
            self.features[4],
        )

        self.encoder3 = nn.Sequential(
            self.features[5],
            self.features[6],
        )

        self.encoder4 = nn.Sequential(
            self.features[7],
            self.features[8]
        )

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

        stm = self.stem(x)
        
        enc1 = self.encoder1(stm)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = self.encoder5(enc4, enc3)

        dec3 = self.decoder4(dec4, enc2)
        dec2 = self.decoder3(dec3, enc1)
        dec1 = self.decoder2(dec2, stm)
        dec0 = self.decoder1(dec1, enc0)
        out = self.out(dec0)
        return out

    def encode(self, x):
        stm = self.stem(x)
        enc1 = self.encoder1(stm)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        return enc4



class Backbone(nn.Module):
    def __init__(
            self,
            in_channels=1,
            feature_size=24,
            patch_size=2,
            stochastic_depth_prob=0.4,
            num_heads=[3, 6, 12, 24],
            block_setting=None,
    ):
        super().__init__()
        
        block = ConvSwinTransformerBlock_A0

        # stem
        firstconv_output_channels = block_setting[0].input_channels
        stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                firstconv_output_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            LayerNorm(firstconv_output_channels, data_format='channels_first'),
        )

        layers = []
        layers.append(stem)

        total_stage_blocks = sum(cnf.num_layers * len(cnf.dilations) for cnf in block_setting)
        stage_block_id = 0
        for i, cnf in enumerate(block_setting):
            # stage
            stage = []
            for _ in range(cnf.num_layers):
                # cal stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage_block_id += 1
                stage.append(block(
                    cnf.input_channels, 
                    sd_prob, 
                    cnf.kernel_sizes[0], 
                    cnf.dilations[0],
                    num_heads[i]
                ))
            layers.append(nn.Sequential(*stage))
            
            # Downsampling
            if cnf.out_channels is not None:
                layers.append(
                    nn.Sequential(
                        LayerNorm(cnf.input_channels, data_format='channels_first'),
                        nn.Conv3d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )
        
        # make features
        self.features = nn.Sequential(*layers)
        
        # init weight
        self.apply(self._init_weights)

    def forward(self, input):
        return self.features(input)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


