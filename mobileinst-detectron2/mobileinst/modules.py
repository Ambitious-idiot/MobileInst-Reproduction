import torch
from torch import nn
import math
from fvcore.nn.weight_init import c2_xavier_fill
from detectron2.layers import NaiveSyncBatchNorm, FrozenBatchNorm2d
from typing import Literal


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bn_weight_init: float = 1,
                 norm: Literal['BN', 'FrozenBN', 'SyncBN'] = 'BN'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups)
        c2_xavier_fill(conv)
        self.add_module('c', conv)

        if norm == "FrozenBN":
            bn = FrozenBatchNorm2d(out_channels)
        elif norm == "SyncBN":
            bn = NaiveSyncBatchNorm(out_channels)
        else:
            bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class BaseAttention(nn.Module):
    def __init__(self, dim, num_kernels, key_dim, num_heads, attn_ratio=4,
                 activation=nn.GELU, norm='bn'):
        super(BaseAttention, self).__init__()
        self.dim = dim
        self.num_kernels = num_kernels
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.scale = key_dim ** -0.5
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.to_q = nn.Sequential(nn.Linear(dim, self.nh_kd), nn.BatchNorm1d(self.nh_kd))
        self.to_k = Conv2d_BN(dim, self.nh_kd, norm=norm)
        self.to_v = Conv2d_BN(dim, self.dh, norm=norm)
        self.proj = nn.Sequential(activation(), nn.Linear(self.dh, dim), nn.BatchNorm1d(dim))

    def _attention(self, q, k, v, B):
        # Q: B*NH*NK*KD K: B*NH*KD*HW V: B*NH*HW*D Out: B*NK*C
        attn = torch.matmul(q, k)
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, v)
        x = x.permute(0, 1, 3, 2).reshape(B, self.dh, self.num_kernels).permute(0, 2, 1).reshape(-1, self.dh)
        return self.proj(x).reshape(-1, self.num_kernels, self.dim)

    def attention(self, q, k, v, B):
        # Q: B*NK*C K: B*C*H*W V: B*C*H*W Out: B*NK*C
        Q = q.reshape(-1, self.dim)
        Q = self.to_q(Q).reshape(B, self.num_kernels, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        K = self.to_k(k).reshape(B, self.num_heads, self.key_dim, -1)
        V = self.to_v(v).reshape(B, self.num_heads, self.d, -1).permute(0, 1, 3, 2)
        return self._attention(Q, K, V, B) + q


class FixedQAttention(BaseAttention):
    def __init__(self, dim, num_kernels, key_dim, num_heads, attn_ratio=4,
                 activation=nn.GELU, norm='bn'):
        super(FixedQAttention, self).__init__(dim, num_kernels, key_dim, num_heads, attn_ratio, activation, norm)
        self.q = nn.Parameter(torch.randn(1, num_kernels, dim))

    def forward(self, x):
        B, *_ = get_shape(x)
        q = self.q.repeat(B, 1, 1)
        return self.attention(q, x, x, B)


class CrossAttention(BaseAttention):
    def __init__(self, dim, num_kernels, key_dim, num_heads, attn_ratio=4,
                 activation=nn.GELU, norm='bn'):
        super(CrossAttention, self).__init__(dim, num_kernels, key_dim, num_heads, attn_ratio, activation, norm)

    def forward(self, x, y):
        B, *_ = get_shape(x)
        return self.attention(y, x, x, B)


class SelfAttention(BaseAttention):
    def __init__(self, dim, num_kernels, key_dim, num_heads, attn_ratio=4,
                 activation=nn.GELU, norm='bn'):
        super(SelfAttention, self).__init__(dim, num_kernels, key_dim, num_heads, attn_ratio, activation, norm)
        self.to_k = nn.Sequential(nn.Linear(dim, self.nh_kd), nn.BatchNorm1d(self.nh_kd))
        self.to_v = nn.Sequential(nn.Linear(dim, self.dh), nn.BatchNorm1d(self.dh))

    def attention(self, q, k, v, B):
        # Q: B*N*C K: B*N*C V: B*N*C Out: B*N*C
        Q = q.reshape(-1, self.dim)
        Q = self.to_q(Q).reshape(B, self.num_kernels, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        K = k.reshape(-1, self.dim)
        K = self.to_k(K).reshape(B, self.num_kernels, self.num_heads, self.key_dim).permute(0, 2, 3, 1)
        V = v.reshape(-1, self.dim)
        V = self.to_v(V).reshape(B, self.num_kernels, self.num_heads, self.d).permute(0, 2, 1, 3)
        return self._attention(Q, K, V, B) + q

    def forward(self, x):
        B, *_ = get_shape(x)
        return self.attention(x, x, x, B)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
