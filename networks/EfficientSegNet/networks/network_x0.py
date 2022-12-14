from functools import partial

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock, UnetrPrUpBlock

from networks.EfficientSegNet.blocks.residual_block import ResBaseConvBlock


class EfficientSegNet_X0(nn.Module):
  def __init__(self, in_channels, out_channels, feature_size=8):
    super().__init__()
    encoder_conv_block = ResBaseConvBlock
    spatial_dims = 3
    norm_name='instance'
    num_layers = 5
    num_channel = [feature_size*(2**i) for i in range(num_layers)]
    num_blocks = [2 for _ in range(num_layers)]
    self.encoder1 = self._mask_layer(encoder_conv_block, 1, num_channel[0], num_blocks[0], stride=1)
    self.encoder2 = self._mask_layer(encoder_conv_block, num_channel[0], num_channel[1], num_blocks[0], stride=2)
    self.encoder3 = self._mask_layer(encoder_conv_block, num_channel[1], num_channel[2], num_blocks[1], stride=2)
    self.encoder4 = self._mask_layer(encoder_conv_block, num_channel[2], num_channel[3], num_blocks[2], stride=2)
    self.encoder5 = self._mask_layer(encoder_conv_block, num_channel[3], num_channel[4], num_blocks[3], stride=2)

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
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )

    self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

  def _mask_layer(self, block, in_channels, out_channels, num_block, stride):
        layers = []
        layers.append(block(in_channels, out_channels, p=0.2, stride=stride, is_identify=False))
        for _ in range(num_block-1):
            layers.append(block(out_channels, out_channels, p=0.2, stride=1, is_identify=True))

        return nn.Sequential(*layers)

  def forward(self,x):
    enc1 = self.encoder1(x)
    enc2 = self.encoder2(enc1)
    enc3 = self.encoder3(enc2)
    enc4 = self.encoder4(enc3)
    enc5 = self.encoder5(enc4)

    dec4 = self.decoder4(enc5, enc4)
    dec3 = self.decoder3(dec4, enc3)
    dec2 = self.decoder2(dec3, enc2)
    dec1 = self.decoder1(dec2, enc1)
    out = self.out(dec1)
    
    return out
        