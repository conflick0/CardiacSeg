import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.stochastic_depth import StochasticDepth

from timm.models.layers import trunc_normal_

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

from .blocks.mnx import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlocks
from .blocks.utils import LayerNorm


class UNETCNX_A3(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            patch_size=4,
            stochastic_depth_prob=0.0,
            feature_size=48,
            depths=[2, 2, 2, 2, 2],
            use_init_weights=False,
            **kwargs
    ) -> None:
        super().__init__()
        
        kernel_size = 5
        exp_rate = 2
        feature_sizes = [feature_size*(2**i) for i in range(len(depths))]
        
        self.backbone = Backbone(
            in_channels=in_channels,
            feature_sizes=feature_sizes,
            depths=depths,
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            stochastic_depth_prob=stochastic_depth_prob,
            use_init_weights=use_init_weights,
        )
        
        self.decoder4 = MedNeXtUpBlocks(
            dim=feature_sizes[3],
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            depths=2,
        )

        self.decoder3 = MedNeXtUpBlocks(
            dim=feature_sizes[2],
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            depths=2,
        )

        self.decoder2 =MedNeXtUpBlocks(
            dim=feature_sizes[1],
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            depths=2,
        )

        self.decoder1 = MedNeXtUpBlocks(
            dim=feature_sizes[0],
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            depths=2,
        )

        self.out_block = UnetOutBlock(spatial_dims=3, in_channels=feature_sizes[0], out_channels=out_channels)

    def forward(self, x):
        hidden_states_out = self.backbone(x)
        
        enc1 = hidden_states_out[0]
        enc2 = hidden_states_out[1]
        enc3 = hidden_states_out[2]
        enc4 = hidden_states_out[3]
        enc5 = hidden_states_out[4]

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
        feature_sizes=[48, 96, 192, 384],
        depths=[3, 3, 9, 3],
        kernel_size=7,
        dilation=1,
        exp_rate=4,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6,
        use_init_weights=False,
    ):
        super().__init__()
        
        print('feature sizes:', feature_sizes)
        print('depths:', depths)
        print('drop rate:',stochastic_depth_prob)
        print('use init weights:', use_init_weights)
        
        stages = []
        for i, (depth, feature_size) in enumerate(zip(depths, feature_sizes)):
            stage = []
            
            if i == 0:
                # stem
                stage.append(nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        feature_size,
                        kernel_size=1,
                    ),
                ))
            else:
                # down sample
                stage.append(MedNeXtDownBlock(
                    dim=feature_size//2,
                    kernel_size=kernel_size,
                    exp_rate=exp_rate,
                    stride=2
                ))
            
            # basic block
            for i in range(depth):
                stage.append(MedNeXtBlock(
                    dim=feature_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    exp_rate=exp_rate,
                    stochastic_depth_prob=0.0,
                    layer_scale=layer_scale
                ))
                
            stages.append(nn.Sequential(*stage))
        
        # build stages
        self.stages = nn.Sequential(*stages)
        
        self.depths = depths
        
        if use_init_weights:
            # init weight
            print('use init weights')
            self.apply(self._init_weights)

    def forward(self, x):
        outs = []
        for i in range(len(self.depths)):
            # forward feature
            x = self.stages[i](x)
            outs.append(x)
            
        return outs
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


