import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision import models
from torchviz import make_dot, make_dot_from_trace
import torch.nn.functional as F

from thop import profile
import time
import matplotlib.pyplot as plt
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict


from model.coordatt import CoordAtt
from model.Fusion_Module_02 import Fusion_Module_02
# from coordatt import CoordAtt
# from Fusion_Module_02 import Fusion_Module_02


import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out



class VSSBlock_new(nn.Module):
    def __init__(
                    self,
                    hidden_dim: int = 0,
                    drop_path: float = 0,
                    norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                    attn_drop_rate: float = 0,
                    d_state: int = 16,
                    **kwargs,
                ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        
        B, C, H, W = input.size()

        x = input.reshape(B, C, H*W).permute(0, 2, 1)
        x = self.ln_1(x.reshape(B, H*W, -1))
        x = x.reshape(B, H, W, C)
        x = self.self_attention(x)
        x = self.drop_path(x)
        x = x.permute(0, 3, 1, 2)

        # print(input.shape)
        # print(x.shape)

        x = input + x

        # print(x.size())

        return x




class Res_CA_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CA_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.CA = CoordAtt(in_channels, out_channels)


    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.CA(out)
        out += residual
        out = self.relu(out)
        return out










class dw_conv(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True):
        super(dw_conv, self).__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.dw_conv_k3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            activation())
    def forward(self, x):
        x = self.dw_conv_k3(x)
        return x

# Initial Downsampling
class InitialBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x) #1 8 128 128
        ext = self.ext_branch(x) #1 3 128 128

        # Concatenate branches
        out = torch.cat((main, ext), 1) # 1 16 128 128

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)

# Regular/asymmetric/depthwise/dilated conv.
class RegularBottleneck(nn.Module):
    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 depthwise=False,
                 dilated=False,
                 regular=False,
                 dropout_prob=0.0,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        if asymmetric:
            internal_channels = channels // internal_ratio
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif depthwise:
            internal_channels = channels * 2
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = dw_conv(internal_channels,internal_channels) #深度可分离卷积

            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif dilated:
            internal_channels = channels // internal_ratio
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif regular:
            internal_channels = channels // internal_ratio
            # 1x1 projection convolution
            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            # 1x1 expansion convolution
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)

# Middle Downsampling
class DownsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0.0,
                 bias=False,
                 relu=True):
        super().__init__()


        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2)

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut

        main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)

# Lightweight Infrared small segmentation
class LW_IRST_ablation_CA_Fuse_mamba_new(nn.Module):

    def __init__(self, n_classes=1, encoder_relu=False, decoder_relu=True, channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3)):
       
        super().__init__()

        # Stage 1 - Encoder
        self.initial_block = InitialBlock(3, channel[0], relu=encoder_relu)

        self.CA1 = CoordAtt(channel[0], channel[0])
        self.vss1 = VSSBlock_new(channel[0])
        self.fuse_01 = Fusion_Module_02(channel[0], channel[0])

        # Stage 2 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            channel[0],
            channel[1],
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)

        self.CA2 = CoordAtt(channel[1], channel[1])
        self.vss2 = VSSBlock_new(channel[1])
        self.fuse_02 = Fusion_Module_02(channel[1], channel[1])

        # Stage 3 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            channel[1],
            channel[2],
            dropout_prob=0.1,
            relu=encoder_relu)
        #DAAA Module1
        self.Depthwise2_1 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous2_2 = RegularBottleneck(
            channel[2], dilation=dilations[0], padding=dilations[0], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric2_3 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[0],
            padding=padding[0],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous2_4 = RegularBottleneck(
            channel[2], dilation=dilations[1], padding=dilations[1], dilated=True, dropout_prob=0.1, relu=encoder_relu)

        # DAAA Module2
        self.Depthwise2_5 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous2_6 = RegularBottleneck(
            channel[2], dilation=dilations[2], padding=dilations[2], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric2_7 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[1],
            padding=padding[1],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous2_8 = RegularBottleneck(
            channel[2], dilation=dilations[3], padding=dilations[3], dilated=True, dropout_prob=0.1, relu=encoder_relu)

        # DAAA Module3
        self.Depthwise3_1 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous3_2 = RegularBottleneck(
            channel[2], dilation=dilations[0], padding=dilations[0], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric3_3 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[2],
            padding=padding[2],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous3_4 = RegularBottleneck(
            channel[2], dilation=dilations[1], padding=dilations[1], dilated=True, dropout_prob=0.1, relu=encoder_relu)

        # DAAA Module4
        self.Depthwise3_5 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous3_6 = RegularBottleneck(
            channel[2], dilation=dilations[2], padding=dilations[2], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric3_7 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[3],
            padding=padding[3],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous3_8 = RegularBottleneck(
            channel[2], dilation=dilations[3], padding=dilations[3], dilated=True, dropout_prob=0.1, relu=encoder_relu)

        self.CA3 = CoordAtt(channel[2], channel[2])
        self.vss3 = VSSBlock_new(channel[2])


        # Stage 4 - Decoder
        self.transposed4_conv = nn.ConvTranspose2d(
            channel[2],
            channel[1],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

        # Stage 5 - Decoder
        self.regular5_1 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_2 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.1, relu=decoder_relu)

        self.transposed5_conv = nn.ConvTranspose2d(
            channel[1],
            channel[0],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

        # Stage 6 - Decoder
        self.regular6 = RegularBottleneck(
            channel[0], padding=1, regular=True, dropout_prob=0.1, relu=decoder_relu)
        self.transposed6_conv = nn.ConvTranspose2d(
            in_channels = channel[0],
            out_channels = n_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

        '''
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                stride=1,
                bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                16,
                8,
                kernel_size=1,
                stride=1,
                bias=False), nn.BatchNorm2d(8), nn.ReLU())

        self.conv1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(8, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        '''


        self.conv_1x1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, bias=False)


    def forward(self, x):

        # Stage 1-Encoder
        input_size = x.size()         # 1 3 256 256
        x1 = self.initial_block(x)    # 1 8 128 128
        x1 = self.vss1(x1)
        stage1_input_size = x1.size() # 1 8 128 128

        # Stage 2-Encoder
        x2 = self.downsample1_0(x1)   # 1 32 64 64
        x2 = self.regular1_1(x2)      # 1 32 64 64
        x2 = self.regular1_2(x2)      # 1 32 64 64
        x2 = self.regular1_3(x2)      # 1 32 64 64
        x2 = self.regular1_4(x2)      # 1 32 64 64

        x2 = self.vss2(x2)

        stage2_input_size = x2.size() # 1 32 64 64

        # Stage3.1 -Encoder
        x3 = self.downsample2_0(x2)   # 1 64 32 32
        #DAAA Module1
        x3 = self.Depthwise2_1(x3)    # 1 64 32 32
        x3 = self.Atrous2_2(x3)       # 1 64 32 32
        x3 = self.Asymmetric2_3(x3)   # 1 64 32 32
        x3 = self.Atrous2_4(x3)       # 1 64 32 32
        #DAAA Module2
        x3 = self.Depthwise2_5(x3)    # 1 64 32 32
        x3 = self.Atrous2_6(x3)       # 1 64 32 32
        x3 = self.Asymmetric2_7(x3)   # 1 64 32 32
        x3 = self.Atrous2_8(x3)       # 1 64 32 32

        # Stage3.2 -Encoder
        #DAAA Module3
        x3 = self.Depthwise3_1(x3)    # 1 64 32 32
        x3 = self.Atrous3_2(x3)       # 1 64 32 32
        x3 = self.Asymmetric3_3(x3)   # 1 64 32 32
        x3 = self.Atrous3_4(x3)       # 1 64 32 32
        #DAAA Module4
        x3 = self.Depthwise3_5(x3)    # 1 64 32 32
        x3 = self.Atrous3_6(x3)       # 1 64 32 32
        x3 = self.Asymmetric3_7(x3)   # 1 64 32 32
        x3 = self.Atrous3_8(x3)       # 1 64 32 32

        x3 = self.vss3(x3)

        # Stage4 -Decoder
        x4 = self.transposed4_conv(x3)                                  # 1 32 64 64

        x4 = self.CA2(x4)
        # fuse
        x4 = self.fuse_02(x2, x4)
        # sum
        x4 = x4 + x2                                                    # 1 32 64 64
        # concat
        # x4 = torch.cat([x4, x2], dim=1)
        # x4 = self.ext_conv1(x4)
        # x4 = self.regular4_1(x4)                                      # 1 32 64 64
        # x4 = self.regular4_2(x4)                                      # 1 32 64 64

        # Stage5 -Decoder
        x5 = self.regular5_1(x4)                                        # 1 32 64 64
        x5 = self.regular5_2(x5)                                        # 1 32 64 64
        x5 = self.transposed5_conv(x5)                                  # 1 8 128 128

        x5 = self.CA1(x5)
        # fuse
        x5 = self.fuse_01(x1, x5)
        # sum
        x5 = x5 + x1                                                    # 1 8 128 128
        # concat
        # x5 = torch.cat([x5, x1], dim=1)
        # x5 = self.ext_conv2(x5)

        # Stage6 -Decoder
        x6 = self.regular6(x5)
        x6 = self.transposed6_conv(x6)                                  # 1 1 256 256
        # x6 = self.conv3(x5)  # 1 16 128 128  1*1conv.
        # x6 = F.interpolate(x6, size=(256, 256), mode='bilinear', align_corners=True)


        # 8邻域聚类
        x6 = self.conv_1x1(x6)                                          # 1 16 128 128  1*1conv.

        return x6



if __name__ == '__main__':

    inputs = torch.randn((1, 3, 256, 256)).cuda()
    model = LW_IRST_ablation_CA_Fuse_mamba_new(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3)).cuda() # kernel_size/padding = 5/2 7/3 9/4

    '''
    start = time.perf_counter()
    out = model(inputs)
    end = time.perf_counter()
    running_FPS = 1 / (end - start)

    FLOPs, params = profile(model, inputs=(inputs,))

    # 可视化
    graph = make_dot(out, params=dict(model.named_parameters()))

    # 优化输出格式
    print(f'running_FPS: {running_FPS}')
    print(f'FLOPs: {FLOPs/1e6:.2f}M')
    print(f'params: {params/1e6:.2f}M')
    print(f'Output size: {out.size()}')
    '''

    start = time.perf_counter()
    for i in range(2000):
        out = model(inputs)
    end = time.perf_counter()

    running_FPS = 2000 / (end - start)
    running_time = 1 / running_FPS


    FLOPs, params = profile(model, inputs=(inputs,))

    # 可视化
    graph = make_dot(out, params=dict(model.named_parameters()))

    # 优化输出格式
    print(f'running_FPS: {running_FPS}')
    print(f'running_time: {running_time}')

    print(f'FLOPs: {FLOPs/1e6:.2f}M')
    print(f'params: {params/1e6:.2f}M')
    print(f'Output size: {out.size()}')