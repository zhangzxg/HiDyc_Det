import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import DropPath
from torch import Tensor
from ultralytics.nn.modules.block import C3k, C3k2



__all__ = ['PHCA_Block', 'C3k_PHCA', 'C3k2_PHCA']

class C3k_PHCA(C3k):
    def __init__(self, c1, c2, n=1, stage=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PHCA_Block(c_, stage) for _ in range(n)))

class C3k2_PHCA(C3k2):
    def __init__(self, c1, c2, n=1, stage=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_PHCA(self.c, self.c, 2, stage, shortcut, g) if c3k else PHCA_Block(self.c, stage) for _ in range(n))



def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        raise NotImplementedError(f'Pad type [{pad_type}] is not recognized.')
    return PadLayer


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super().__init__()

        self.filt_size = filt_size
        self.pad_off = pad_off
        self.stride = stride
        self.channels = channels

        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2))
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]

        if self.filt_size == 1:
            a = np.array([1.0])
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        else:
            raise NotImplementedError(f'filt_size [{self.filt_size}] is not supported.')

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)

        self.register_buffer(
            'filt',
            filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]

        return F.conv2d(
            self.pad(inp),
            self.filt,
            stride=self.stride,
            groups=inp.shape[1]
        )


class DetailGatedAttention(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()

        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, bias=False),
            norm_layer(dim * 4),
            act_layer(),
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False)
        )

        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        out = x * self.gate_fn(att)
        return out



class LocalPerceptionAttention(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        out = self.conv(x)
        return out



class RegionalDilatedContextModule(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()

        att_padding = att_kernel // 2

        self.channel = channel
        self.gate_fn = nn.Sigmoid()

        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = BlurPool(channel, stride=2)

        self.H_att1 = nn.Conv2d(
            channel,
            channel,
            kernel_size=(att_kernel, 3),
            stride=1,
            padding=(att_padding, 1),
            groups=channel,
            bias=False
        )

        self.V_att1 = nn.Conv2d(
            channel,
            channel,
            kernel_size=(3, att_kernel),
            stride=1,
            padding=(1, att_padding),
            groups=channel,
            bias=False
        )

        self.H_att2 = nn.Conv2d(
            channel,
            channel,
            kernel_size=(att_kernel, 3),
            stride=1,
            padding=(att_padding, 1),
            groups=channel,
            bias=False
        )

        self.V_att2 = nn.Conv2d(
            channel,
            channel,
            kernel_size=(3, att_kernel),
            stride=1,
            padding=(1, att_padding),
            groups=channel,
            bias=False
        )

        self.dilated_context = nn.Conv2d(
            channel,
            channel,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            groups=channel,
            bias=False
        )

        self.dilate_gamma = nn.Parameter(torch.tensor(0.0))

        self.norm = norm_layer(channel)

    def forward(self, x):
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)

        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)

        x_h2 = self.inv_h_transform(
            self.H_att2(self.h_transform(x_tem))
        )

        x_w2 = self.inv_v_transform(
            self.V_att2(self.v_transform(x_tem))
        )

        x_d = self.dilated_context(x_tem)

        att = self.norm(
            x_h1 + x_w1 + x_h2 + x_w2 + self.dilate_gamma * x_d
        )

        out = x[:, :self.channel, :, :] * F.interpolate(
            self.gate_fn(att),
            size=(x.shape[-2], x.shape[-1]),
            mode='nearest'
        )

        return out

    def h_transform(self, x):
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)



class GlobalSelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 head_dim=4,
                 num_heads=None,
                 qkv_bias=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 proj_bias=False):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)
        N = H * W

        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(
            B, H, W, self.attention_dim
        )

        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.permute(0, 3, 1, 2)

        return x


class DownsampledGlobalAttention(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()

        self.norm = norm_layer(dim)
        self.attn = GlobalSelfAttention(dim)

        self.downpool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=True
        )

        self.uppool = nn.MaxUnpool2d(
            kernel_size=(2, 2),
            stride=2,
            padding=0
        )

    def forward(self, x):
        x_, idx = self.downpool(x)
        x_ = self.norm(self.attn(x_))
        x = self.uppool(x_, indices=idx)
        return x


class PoolGuidedGlobalMixer(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()

        self.downpool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=True
        )

        self.uppool = nn.MaxUnpool2d(
            kernel_size=(2, 2),
            stride=2,
            padding=0
        )

        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = act_layer()

        self.conv0 = nn.Conv2d(
            dim,
            dim,
            kernel_size=5,
            padding=2,
            groups=dim
        )

        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            stride=1,
            padding=9,
            groups=dim,
            dilation=3
        )

        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim // 2, kernel_size=1)

        self.conv_squeeze = nn.Conv2d(2, 2, kernel_size=7, padding=3)

        self.conv = nn.Conv2d(dim // 2, dim, kernel_size=1)
        self.proj_2 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x_, idx = self.downpool(x)

        x_ = self.proj_1(x_)
        x_ = self.activation(x_)

        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)

        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)

        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        attn = (
            attn1 * sig[:, 0, :, :].unsqueeze(1) +
            attn2 * sig[:, 1, :, :].unsqueeze(1)
        )

        attn = self.conv(attn)

        x_ = x_ * attn
        x_ = self.proj_2(x_)

        x = self.uppool(x_, indices=idx)

        return x


class StageAwareGlobalMixer(nn.Module):
    def __init__(self, dim, stage, norm_layer, act_layer):
        super().__init__()

        self.stage = stage

        if stage == 2:
            self.global_mixer = DownsampledGlobalAttention(dim, norm_layer)
            self.norm = None
        elif stage == 3:
            self.global_mixer = GlobalSelfAttention(dim)
            self.norm = norm_layer(dim)
        else:
            self.global_mixer = PoolGuidedGlobalMixer(dim, act_layer)
            self.norm = norm_layer(dim)

    def forward(self, x):
        if self.stage == 2:
            out = x + self.global_mixer(x)
        else:
            out = self.norm(x + self.global_mixer(x))

        return out


class ScaleAwareBranchFusion(nn.Module):
    def __init__(self, dim, branch_num=4, reduction=16, act_layer=nn.SiLU):
        super().__init__()

        hidden_dim = max(dim // reduction, branch_num)

        self.branch_num = branch_num

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(hidden_dim, branch_num, kernel_size=1, bias=True)
        )

        self.softmax = nn.Softmax(dim=1)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, xs):
        x_cat = torch.cat(xs, dim=1)

        descriptor = self.avg_pool(x_cat) + self.max_pool(x_cat)

        weight = self.softmax(self.fc(descriptor))

        scale = 1.0 + self.gamma * (weight * self.branch_num - 1.0)

        out = torch.cat(
            [xs[i] * scale[:, i:i + 1, :, :] for i in range(self.branch_num)],
            dim=1
        )

        return out


class PHCA_Block(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 att_kernel=11,
                 mlp_ratio=2.0,
                 drop_path=0.0,
                 act_layer=nn.SiLU,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        assert dim % 4 == 0, f'PHCA_Block requires dim to be divisible by 4, but got dim={dim}.'

        self.stage = stage
        self.dim_split = dim // 4

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, kernel_size=1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, kernel_size=1, bias=False)
        )

        self.dga = DetailGatedAttention(
            self.dim_split,
            norm_layer,
            act_layer
        )

        self.lpa = LocalPerceptionAttention(
            self.dim_split,
            norm_layer,
            act_layer
        )

        self.rdcm = RegionalDilatedContextModule(
            self.dim_split,
            att_kernel,
            norm_layer
        )

        self.sagm = StageAwareGlobalMixer(
            self.dim_split,
            stage,
            norm_layer,
            act_layer
        )
        self.local_alpha = nn.Parameter(torch.tensor(0.0))
        self.context_alpha = nn.Parameter(torch.tensor(0.0))

        self.branch_fusion = ScaleAwareBranchFusion(
            dim=dim,
            branch_num=4,
            reduction=16,
            act_layer=act_layer
        )

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        x1, x2, x3, x4 = torch.split(
            x,
            [self.dim_split, self.dim_split, self.dim_split, self.dim_split],
            dim=1
        )
        x1 = x1 + self.dga(x1)

        x2 = self.lpa(x2) + self.local_alpha * x2

        x3 = self.rdcm(x3) + self.context_alpha * x3

        x4 = self.sagm(x4)
        x_att = self.branch_fusion([x1, x2, x3, x4])

        out = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))

        return out
