import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.stochastic_depth import StochasticDepth

from timm.models.layers import trunc_normal_

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

from .blocks.convnext import DiConvNeXt
from .blocks.utils import LayerNorm


class UNETCNX_A2(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            patch_size=4,
            spatial_dims=3,
            norm_name='instance',
            stochastic_depth_prob=0.1,
            feature_size=48,
            depths=[3, 3, 9, 3],
            use_init_weights=False,
            res_block=True,
            **kwargs
    ) -> None:
        super().__init__()
        
        feature_sizes = [feature_size*(2**i) for i in range(len(depths))]
        print(feature_sizes)
        
        self.backbone = Backbone(
            in_channels=in_channels,
            patch_size=patch_size,
            feature_sizes=feature_sizes,
            depths=depths,
            kernel_size=7,
            dilation=1,
            stochastic_depth_prob=stochastic_depth_prob,
            use_init_weights=use_init_weights,
        )
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_sizes[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[0],
            out_channels=feature_sizes[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[1],
            out_channels=feature_sizes[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[2],
            out_channels=feature_sizes[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3],
            out_channels=feature_sizes[3]*2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3]*2,
            out_channels=feature_sizes[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3],
            out_channels=feature_sizes[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[2],
            out_channels=feature_sizes[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[1],
            out_channels=feature_sizes[0],
            kernel_size=3,
            upsample_kernel_size=patch_size,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out_block = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_sizes[0], out_channels=out_channels)

    def forward(self, x):
        enc1 = self.encoder1(x)
        
        hidden_states_out = self.backbone(x)
        
        enc2 = self.encoder2(hidden_states_out[0])
        enc3 = self.encoder3(hidden_states_out[1])
        enc4 = self.encoder4(hidden_states_out[2])
        enc5 = self.encoder5(hidden_states_out[3])

        dec4 = self.decoder4(enc5, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        
        out = self.out_block(dec1)
        return out

    def encode(self, x):
        hidden_states_out = self.backbone(x)
        return hidden_states_out[3]


class Backbone(nn.Module):
    def __init__(
            self,
            in_channels=1,
            patch_size=2,
            feature_sizes=[48, 96, 192, 384],
            depths=[3, 3, 9, 3],
            kernel_size=7,
            dilation=1,
            stochastic_depth_prob=0.0,
            use_init_weights=False,
    ):
        super().__init__()
        
        print('patch size:', patch_size)
        print('feature sizes:', feature_sizes)
        print('depths:', depths)
        print('drop rate:',stochastic_depth_prob)
        print('use init weights:', use_init_weights)
        
        block = DiConvNeXt

        stages = []
        # build stages 
        total_stage_blocks = sum(d for d in depths)
        stage_block_id = 0
        for i, (depth, feature_size) in enumerate(zip(depths, feature_sizes)):
            # build one stage by downsample block and blocks 
            stage = []
            
            # add downsample(stem) block
            if i == 0:
                # stem block
                stage.append(nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        feature_size,
                        kernel_size=patch_size,
                        stride=patch_size,
                        padding=0,
                    ),
                    LayerNorm(feature_size, data_format='channels_first'),
                ))
            else:
                # down sample block
                stage.append(
                    nn.Sequential(
                        LayerNorm(feature_size//2, data_format='channels_first'),
                        nn.Conv3d(feature_size//2, feature_size, kernel_size=2, stride=2),
                    )
                )

            # add blocks to stage
            for _ in range(depth):
                # cal stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage_block_id += 1
                stage.append(
                    block(
                        feature_size, 
                        sd_prob, 
                        kernel_size, 
                        dilation
                    )
                )
            
            # add stage to stages
            stages.append(nn.Sequential(*stage))
        
        # build stages
        self.stages = nn.Sequential(*stages)
        
        if use_init_weights:
            # init weight
            print('use init weights')
            self.apply(self._init_weights)

    def forward(self, x):
        outs = []
        for i in range(4):
            # forward feature
            x = self.stages[i](x)
            outs.append(x)
            
        return outs
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


