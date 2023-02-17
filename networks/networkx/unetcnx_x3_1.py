import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.stochastic_depth import StochasticDepth

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock

from .blocks.utils import LayerNorm, Permute, BlockConfig


class UNETCNX_X3_1(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            feature_size=24,
            patch_size=4,
            spatial_dims=3,
            norm_name='instance',
            stochastic_depth_prob=0.4,
            layer_scale=1e-6,
            **kwargs
    ) -> None:
        super().__init__()
        block_setting = [
            BlockConfig(feature_size * 2, feature_size * 4, 10),
            BlockConfig(feature_size * 4, feature_size * 8, 10),
            BlockConfig(feature_size * 8, feature_size * 16, 18),
            BlockConfig(feature_size * 16, feature_size * 32, 10)
        ]

        self.features = Net(
            in_channels=in_channels,
            feature_size=feature_size,
            patch_size=patch_size,
            stochastic_depth_prob=stochastic_depth_prob,
            layer_scale=layer_scale,
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

        self.inc = self.features[0]

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



class Net(nn.Module):
    def __init__(
            self,
            in_channels=1,
            feature_size=24,
            patch_size=2,
            stochastic_depth_prob=0.4,
            layer_scale=1e-6,
            block_setting=None,
    ):
        super().__init__()

        block = Block
        if block_setting is None:
            block_setting = [
                BlockConfig(feature_size * 2, feature_size * 4, 4),
                BlockConfig(feature_size * 4, feature_size * 8, 4),
                BlockConfig(feature_size * 8, feature_size * 16, 8),
                BlockConfig(feature_size * 16, feature_size * 32, 4)
            ]

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

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # stage
            stage = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                if stage_block_id % 2 == 0:
                    # odd block is conv block
                    stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                else:
                    # even block is dil conv block
                    stage.append(block(cnf.input_channels, layer_scale, sd_prob, dilation=3))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # Downsampling
            if cnf.out_channels is not None:
                layers.append(
                    nn.Sequential(
                        LayerNorm(cnf.input_channels, data_format='channels_first'),
                        nn.Conv3d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)

    def forward(self, input):
        return self.features(input)


class Block(nn.Module):
    def __init__(
        self,
        dim=48,
        layer_scale=1e-6,
        stochastic_depth_prob=0.0,
        dilation=1,
    ) -> None:
        super().__init__()
        kernel_size = 7 if dilation == 1 else 3
        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=3, groups=dim, bias=True, dilation=dilation),
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

