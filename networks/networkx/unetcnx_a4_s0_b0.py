import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.stochastic_depth import StochasticDepth

from timm.models.layers import trunc_normal_

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock, SimpleASPP

from .blocks.cst import ConvNeXtBlock_V1
from .blocks.utils import LayerNorm

from .blocks.cbam import CBAM


class UNETCNX_A4_S0_B0(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            patch_size=4,
            stochastic_depth_prob=0.0,
            feature_size=48,
            kernel_size=7,
            exp_rate=4,
            depths=[2, 2, 2, 2],
            norm_name='layer', 
            use_init_weights=False,
            **kwargs
    ) -> None:
        super().__init__()
        
        feature_sizes = [feature_size*(2**i) for i in range(len(depths))]
        
        decoder_norm_name = 'instance' 
        res_block = True
        spatial_dims = 3
        
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_sizes[0],
            kernel_size=3,
            stride=1,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        
        self.backbone = Backbone(
            in_channels=in_channels,
            feature_sizes=feature_sizes,
            patch_size=patch_size,
            depths=depths,
            kernel_size=kernel_size,
            exp_rate=exp_rate,
            norm_name=norm_name,
            stochastic_depth_prob=stochastic_depth_prob,
            use_init_weights=use_init_weights,
        )
        
        self.bottleneck = SimpleASPP(
            spatial_dims=3, 
            in_channels=feature_sizes[4], 
            conv_out_channels=feature_sizes[4]//4,
            norm_type='instance'
        )
        
        self.skip_encoder1 = CBAM(feature_sizes[0], reduction=16, kernel_size=7)
        self.skip_encoder2 = CBAM(feature_sizes[1], reduction=16, kernel_size=7)
        self.skip_encoder3 = CBAM(feature_sizes[2], reduction=16, kernel_size=7)
        self.skip_encoder4 = CBAM(feature_sizes[3], reduction=16, kernel_size=7)
        self.skip_encoder5 = CBAM(feature_sizes[4], reduction=16, kernel_size=7)
        
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3]*2,
            out_channels=feature_sizes[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[3],
            out_channels=feature_sizes[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[2],
            out_channels=feature_sizes[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[1],
            out_channels=feature_sizes[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )
        
        self.decoder0 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_sizes[0],
            out_channels=feature_sizes[0],
            kernel_size=3,
            upsample_kernel_size=patch_size,
            norm_name=decoder_norm_name,
            res_block=res_block,
        )


        self.out_block = UnetOutBlock(spatial_dims=3, in_channels=feature_sizes[0], out_channels=out_channels)

    def forward(self, x):
        enc0 = self.encoder0(x)
        
        hidden_states_out = self.backbone(x)
        hidden_states_out[4] = self.bottleneck(hidden_states_out[4])
        
        enc1 = self.skip_encoder1(hidden_states_out[0])
        enc2 = self.skip_encoder2(hidden_states_out[1])
        enc3 = self.skip_encoder3(hidden_states_out[2])
        enc4 = self.skip_encoder4(hidden_states_out[3])
        enc5 = self.skip_encoder5(hidden_states_out[4])
        
        dec4 = self.decoder4(enc5, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        dec0 = self.decoder0(dec1, enc0)

        out = self.out_block(dec0)
        return out

    def encode(self, x):
        hidden_states_out = self.backbone(x)
        return hidden_states_out[3]


class Backbone(nn.Module):
    def __init__(
        self,
        in_channels=1,
        feature_sizes=[48, 96, 192, 384],
        patch_size=4,
        depths=[2, 2, 2, 2],
        kernel_size=7,
        dilation=1,
        exp_rate=4,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6,
        norm_name='layer',
        use_init_weights=False,
    ):
        super().__init__()
        
        print('ker size:', kernel_size)
        print('exp rate:', exp_rate)
        print('feature sizes:', feature_sizes)
        print('depths:', depths)
        print('norm name:', norm_name)
        print('drop rate:',stochastic_depth_prob)
        print('use init weights:', use_init_weights)
        
        stages = []
        total_stage_blocks = sum(d for d in depths)
        stage_block_id = 0
        for i, (depth, feature_size) in enumerate(zip(depths, feature_sizes)):
            stage = []
            
            if i == 0:
                # stem
                stage.append(
                    nn.Sequential(
                        nn.Conv3d(in_channels, feature_size, kernel_size=patch_size, stride=patch_size),
                        _get_norm(norm_name, feature_size),
                    )
                )
            else:
                # down sample
                stage.append(
                    nn.Sequential(
                         _get_norm(norm_name, feature_size//2),
                        nn.Conv3d(feature_size//2, feature_size, kernel_size=2, stride=2),
                    )
                )
                
            # basic block
            for _ in range(depth):
                # cal stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage_block_id += 1

                stage.append(ConvNeXtBlock_V1(
                    dim=feature_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    exp_rate=exp_rate,
                    stochastic_depth_prob=sd_prob,
                    layer_scale=layer_scale,
                    norm_name=norm_name
                ))
                
            stages.append(nn.Sequential(*stage))
        
        # build stages
        self.stages = nn.Sequential(*stages)
        
        if use_init_weights:
            # init weight
            print('use init weights')
            self.apply(self._init_weights)

    def forward(self, x):
        outs = []
        for i in range(len(self.stages)):
            # forward feature
            x = self.stages[i](x)
            outs.append(x)
            
        return outs
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


def _get_norm(norm_name, dim):
    if norm_name == 'layer':
        return LayerNorm(dim, data_format='channels_first')
    elif norm_name == 'group':
        num_group = 4
        return nn.GroupNorm(num_group, dim)
    else:
        raise ValueError(f'invalid norm name: {norm_name}')
        
    