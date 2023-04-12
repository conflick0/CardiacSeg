# https://zhuanlan.zhihu.com/p/555252748
# https://www.sciencedirect.com/science/article/abs/pii/S0031320322002667?via%3Dihub

import math

import torch
from torch import nn
from torch.nn import init

import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, Channel_nums):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool3d(1)  # 最大池化
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)  # C1D 一维卷积
        self.sigmoid = nn.Sigmoid()

    def get_kernel_num(self, C):  # 根据通道数求一维卷积大卷积核大小 odd|t|最近奇数
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max
        F_add_ = F_add.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        out = self.sigmoid(F_add_)
        return out
    
    
class SpatialAttention(nn.Module):
    def __init__(self, Channel_num):
        super(SpatialAttention, self).__init__()
        self.channel = Channel_num
        self.Lambda = 0.6  # separation rate 论文中经过实验发现0.6效果最佳
        self.C_im = self.get_important_channelNum(Channel_num)
        self.C_subim = Channel_num - self.C_im
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm_active = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def get_important_channelNum(self, C):  # 根据通道数以及分离率确定重要通道的数量 even|t|最近偶数
        t = self.Lambda * C
        floor = math.floor(t)
        C_im = floor + floor % 2
        return C_im

    
    def get_im_subim_channels(self, C_im, M): # 根据Channel_Attention_Map得到重要通道以及不重要的通道
        _, topk = torch.topk(M, dim=1, k=C_im)
        important_channels = torch.zeros_like(M)
        subimportant_channels = torch.ones_like(M)
        important_channels = important_channels.scatter(1, topk, 1)
        subimportant_channels = subimportant_channels.scatter(1, topk, 0)
        return important_channels, subimportant_channels

    def get_features(self, im_channels, subim_channels, channel_refined_feature):
        import_features = im_channels * channel_refined_feature
        subimportant_features = subim_channels * channel_refined_feature
        return import_features, subimportant_features

    def forward(self, x, M):
        important_channels, subimportant_channels = self.get_im_subim_channels(self.C_im, M)
        important_features, subimportant_features = self.get_features(important_channels, subimportant_channels, x)

        im_AvgPool = torch.mean(important_features, dim=1, keepdim=True) * (self.channel / self.C_im)
        im_MaxPool, _ = torch.max(important_features, dim=1, keepdim=True)

        subim_AvgPool = torch.mean(subimportant_features, dim=1, keepdim=True) * (self.channel / self.C_subim)
        subim_MaxPool, _ = torch.max(subimportant_features, dim=1, keepdim=True)

        im_x = torch.cat([im_AvgPool, im_MaxPool], dim=1)
        subim_x = torch.cat([subim_AvgPool, subim_MaxPool], dim=1)

        A_S1 = self.norm_active(self.conv(im_x))
        A_S2 = self.norm_active(self.conv(subim_x))

        F1 = important_features * A_S1
        F2 = subimportant_features * A_S2

        refined_feature = F1 + F2

        return refined_feature
    
    
class HAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.channel_attention = ChannelAttention(self.channel)
        self.spatial_attention = SpatialAttention(self.channel)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        residual = x_in
        channel_attention_map = self.channel_attention(x_in)
        channel_refined_feature = channel_attention_map * x_in
        final_refined_feature = self.spatial_attention(channel_refined_feature, channel_attention_map)
        out = self.relu(final_refined_feature + residual)
        return out