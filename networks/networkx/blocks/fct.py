import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.ops.stochastic_depth import StochasticDepth

class Convolutional_Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop
        
        self.layer_q = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, stride_q, padding_q, bias=False, groups=channels),
            nn.ReLU(),
        )
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)

        self.layer_k = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, stride_kv, padding_kv, bias=False, groups=channels),
            nn.ReLU(),
        )
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)

        self.layer_v = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, stride_kv, padding_kv, bias=False, groups=channels),
            nn.ReLU(),
        )
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)
        
        self.attention = nn.MultiheadAttention(embed_dim=channels, 
                                               bias=attention_bias, 
                                               batch_first=True,
                                               dropout=self.proj_drop,
                                               num_heads=self.num_heads)

    def _build_projection(self, x, mode):
        # x shape [batch,channel,size,size]
        # mode:0->q,1->k,2->v,for torch.script can not script str
        
        if mode == 0:
            x1 = self.layer_q(x)
            x1 = x1.permute(0, 2, 3, 4, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 4, 1, 2, 3)
        elif mode == 1:
            x1 = self.layer_k(x)
            x1 = x1.permute(0, 2, 3, 4, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 4, 1, 2, 3)          
        elif mode == 2:
            x1 = self.layer_v(x)
            x1 = x1.permute(0, 2, 3, 4, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 4, 1, 2, 3)      

        return proj

    def get_qkv(self, x):
        q = self._build_projection(x, 0)
        k = self._build_projection(x, 1)
        v = self._build_projection(x, 2)

        return q, k, v

    def forward(self, x):
        q, k, v = self.get_qkv(x)
        q = q.view(q.shape[0], q.shape[1], q.shape[2]*q.shape[3]*q.shape[4])
        k = k.view(k.shape[0], k.shape[1], k.shape[2]*k.shape[3]*k.shape[4])
        v = v.view(v.shape[0], v.shape[1], v.shape[2]*v.shape[3]*v.shape[4])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)
        
        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.cbrt(x1.shape[2]).astype(int), np.cbrt(x1.shape[2]).astype(int), np.cbrt(x1.shape[2]).astype(int))

        return x1

class Transformer(nn.Module):
    def __init__(self,
                 out_channels,
                 num_heads,
                 dpr = 0.0,
                 proj_drop=0.0,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()
        
        self.attention_output = Convolutional_Attention(channels=out_channels,
                                         num_heads=num_heads,
                                         proj_drop=proj_drop,
                                         padding_q=padding_q,
                                         padding_kv=padding_kv,
                                         stride_kv=stride_kv,
                                         stride_q=stride_q,
                                         attention_bias=attention_bias,
                                         )

        self.stochastic_depth = StochasticDepth(dpr,mode='batch')
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        # x1 = self.attention_output(x)
        # x1 = self.stochastic_depth(x1)
        # x2 = self.conv1(x1) + x
        x2 = x
        x3 = x2.permute(0, 2, 3, 4, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 4, 1, 2, 3)
        x3 = self.wide_focus(x3)
        x3 = self.stochastic_depth(x3)

        out = x3 + x2
        return out
    
class Wide_Focus(nn.Module): 
    """
    Wide-Focus module.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels,out_channels, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.layer_dilation2 = nn.Sequential(
            nn.Conv3d(in_channels,out_channels, kernel_size=3, stride=1, padding="same", dilation=2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.layer_dilation3 = nn.Sequential(
            nn.Conv3d(in_channels,out_channels, kernel_size=3, stride=1, padding="same", dilation=3),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(in_channels,out_channels, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(0.1)
        )


    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer_dilation2(x)
        x3 = self.layer_dilation3(x)
        added = x1 + x2 + x3
        x_out = self.layer4(added)
        return x_out


class FCTBlock(nn.Module):
    def __init__(self, dim, att_heads, dpr, use_layernorm=True):
        super().__init__()
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = nn.LayerNorm(dim, eps=1e-5)
        # self.layer1 = nn.Sequential(
        #     nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding="same"),
        #     nn.ReLU()
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding="same"),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        # )
        self.trans = Transformer(dim, att_heads, dpr)

    def forward(self, x):
        if self.use_layernorm:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.layernorm(x)
            x = x.permute(0, 4, 1, 2, 3)
        # x1 = self.layer1(x)
        # x1 = self.layer2(x1)
        x1 = self.trans(x)
        return x1