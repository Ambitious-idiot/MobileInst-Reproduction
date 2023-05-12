import math
import torch
from torch import nn
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_xavier_fill
from typing import Literal
from detectron2.layers import ShapeSpec
from detectron2.layers import NaiveSyncBatchNorm, FrozenBatchNorm2d
from detectron2.modeling import Backbone, BACKBONE_REGISTRY



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., norm='BN'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm=norm)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm=norm)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        ks: int,
        stride: int,
        expand_ratio: int,
        activations = None,
        norm='BN'
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, kernel_size=1, norm=norm))
            layers.append(activations())
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, kernel_size=ks, stride=stride, padding=ks//2, groups=hidden_dim, norm=norm),
            activations(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, kernel_size=1, norm=norm)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TokenPyramidModule(nn.Module):
    def __init__(
        self,
        cfgs,
        out_indices,
        inp_channel=16,
        activation=nn.ReLU,
        norm='BN',
        width_mult=1.):
        super().__init__()
        self.out_indices = out_indices

        self.stem = nn.Sequential(
            Conv2d_BN(3, inp_channel, 3, 2, 1, norm=norm),
            activation()
        )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, norm=norm, activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm='BN'):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm=norm)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm=norm)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm=norm)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm=norm))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)

        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1) # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm='BN'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer, norm=norm)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm=norm)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class BasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                mlp_ratio=4., attn_ratio=2., drop=0., drop_path=0.,
                norm='BN',
                act_layer=None):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm=norm,
                act_layer=act_layer))

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm='BN'
    ) -> None:
        super(InjectionMultiSum, self).__init__()
        self.local_embedding = Conv2d_BN(inp, oup, kernel_size=1, norm=norm)
        self.global_embedding = Conv2d_BN(inp, oup, kernel_size=1, norm=norm)
        self.global_act = Conv2d_BN(inp, oup, kernel_size=1, norm=norm)
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


class InjectionMultiSumCBR(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm='BN'
    ) -> None:
        '''
        local_embedding: conv-bn-relu
        global_embedding: conv-bn-relu
        global_act: conv
        '''
        super(InjectionMultiSumCBR, self).__init__()
        self.local_embedding = Conv2d_BN(inp, oup, kernel_size=1, norm=norm)
        self.global_embedding = Conv2d_BN(inp, oup, kernel_size=1, norm=norm)
        self.global_act = Conv2d_BN(inp, oup, kernel_size=1, norm=norm)
        self.act = h_sigmoid()

        self.out_channels = oup

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        # kernel
        global_act = self.global_act(x_g)
        global_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        # feat_h
        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        out = local_feat * global_act + global_feat
        return out


class FuseBlockSum(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        norm='BN'
    ) -> None:
        super(FuseBlockSum, self).__init__()
        self.fuse1 = Conv2d_BN(inp, oup, kernel_size=1, norm=norm)
        self.fuse2 = Conv2d_BN(inp, oup, kernel_size=1, norm=norm)

        self.out_channels = oup

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        kernel = self.fuse2(x_h)
        feat_h = F.interpolate(kernel, size=(H, W), mode='bilinear', align_corners=False)
        out = inp + feat_h
        return out


class FuseBlockMulti(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        norm='BN'
    ) -> None:
        super(FuseBlockMulti, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.fuse1 = Conv2d_BN(inp, oup, kernel_size=1, norm=norm)
        self.fuse2 = Conv2d_BN(inp, oup, kernel_size=1, norm=norm)
        self.act = h_sigmoid()

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out


SIM_BLOCK = {
    "fuse_sum": FuseBlockSum,
    "fuse_multi": FuseBlockMulti,

    "muli_sum":InjectionMultiSum,
    "muli_sum_cbr":InjectionMultiSumCBR,
}


class Topformer(Backbone):
    def __init__(self, model_cfgs, norm='BN',
                 act_layer=nn.ReLU6, injection_type="muli_sum", injection=True,
                 out_features=None):
        super().__init__()
        self.channels = model_cfgs['channels']
        self.injection = injection
        self.embed_dim = sum(self.channels)
        self.decode_out_indices = model_cfgs['decode_out_indices']
        self.out_channels = model_cfgs['out_channels']

        self.tpm = TokenPyramidModule(cfgs=model_cfgs['cfgs'], out_indices=model_cfgs['embed_out_indice'], norm=norm)
        self.ppa = PyramidPoolAgg(stride=model_cfgs['c2t_stride'])

        dpr = [x.item() for x in torch.linspace(0, model_cfgs['drop_path_rate'], model_cfgs['depths'])]  # stochastic depth decay rule
        self.trans = BasicLayer(
            block_num=model_cfgs['depths'],
            embedding_dim=self.embed_dim,
            key_dim=model_cfgs['key_dim'],
            num_heads=model_cfgs['num_heads'],
            mlp_ratio=model_cfgs['mlp_ratios'],
            attn_ratio=model_cfgs['attn_ratios'],
            drop=0, drop_path=dpr,
            norm=norm, act_layer=act_layer)

        # SemanticInjectionModule
        self.SIM = nn.ModuleList()
        inj_module = SIM_BLOCK[injection_type]
        if self.injection:
            for i in range(len(self.channels)):
                if i in self.decode_out_indices:
                    self.SIM.append(
                        inj_module(self.channels[i], self.out_channels[i], norm=norm, activations=act_layer))
                else:
                    self.SIM.append(nn.Identity())

        out_features_names = ["x3", "x4", "x5", "x6"]
        self._out_feature_strides = dict(zip(out_features_names, [8, 16, 32, 64]))
        channels = self.out_channels if self.injection else self.channels
        self._out_feature_channels = dict(
            zip(out_features_names, channels[1:] + [self.embed_dim]))
        if out_features is None:
            self._out_features = out_features_names
        else:
            self._out_features = out_features

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def size_divisibility(self):
        return 64

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        ouputs = self.tpm(x)
        out = self.ppa(ouputs)
        out = self.trans(out)

        if self.injection:
            xx = out.split(self.channels, dim=1)
            results = []
            for i in range(len(self.channels)):
                if i in self.decode_out_indices:
                    local_tokens = ouputs[i]
                    global_semantics = xx[i]
                    out_ = self.SIM[i](local_tokens, global_semantics)
                    results.append(out_)
            results.append(out)
            return dict(zip(self._out_features, results))
        else:
            ouputs.append(out)
            return dict(zip(self._out_features, ouputs[1:]))


topformer_cfgs = {
    'topformer_base': dict(
        cfgs=[
            # k,  t,  c, s
            [3,   1,  16, 1], # 1/2        0.464K  17.461M
            [3,   4,  32, 2], # 1/4 1      3.44K   64.878M
            [3,   3,  32, 1], #            4.44K   41.772M
            [5,   3,  64, 2], # 1/8 3      6.776K  29.146M
            [5,   3,  64, 1], #            13.16K  30.952M
            [3,   3,  128, 2], # 1/16 5     16.12K  18.369M
            [3,   3,  128, 1], #            41.68K  24.508M
            [5,   6,  160, 2], # 1/32 7     0.129M  36.385M
            [5,   6,  160, 1], #            0.335M  49.298M
            [3,   6,  160, 1], #            0.335M  49.298M
        ],
        channels=[32, 64, 128, 160],
        out_channels=[None, 256, 256, 256],
        embed_out_indice=[2, 4, 6, 9],
        decode_out_indices=[1, 2, 3],
        num_heads=8,
        c2t_stride=2,
        depths=4,
        drop_path_rate=0.1,
        key_dim=16,
        attn_ratios=2,
        mlp_ratios=2
    ),
    'topformer_tiny': dict(
        cfg=[
        # k,  t,  c, s
            [3,   1,  16, 1], # 1/2        0.464K  17.461M
            [3,   4,  16, 2], # 1/4 1      3.44K   64.878M
            [3,   3,  16, 1], #            4.44K   41.772M
            [5,   3,  32, 2], # 1/8 3      6.776K  29.146M
            [5,   3,  32, 1], #            13.16K  30.952M
            [3,   3,  64, 2], # 1/16 5     16.12K  18.369M
            [3,   3,  64, 1], #            41.68K  24.508M
            [5,   6,  96, 2], # 1/32 7     0.129M  36.385M
            [5,   6,  96, 1], #            0.335M  49.298M
        ],
        channels=[16, 32, 64, 96],
        out_channels=[None, 128, 128, 128],
        embed_out_indice=[2, 4, 6, 8],
        decode_out_indices=[1, 2, 3],
        num_heads=4,
        c2t_stride=2,
        depths=4,
        drop_path_rate=0.1,
        key_dim=16,
        attn_ratios=2,
        mlp_ratios=2
    )
}


@BACKBONE_REGISTRY.register()
def build_topformer_backbone(cfg, input_shape):
    name = cfg.MODEL.TOPFORMER.NAME
    injection = cfg.MODEL.TOPFORMER.INJECTION
    norm = cfg.MODEL.TOPFORMER.NORM
    injection_type = cfg.MODEL.TOPFORMER.INJECTION_TYPE
    out_features = cfg.MODEL.TOPFORMER.OUT_FEATURES

    return Topformer(
        model_cfgs=topformer_cfgs[name],
        norm=norm,
        injection_type=injection_type,
        injection=injection,
        out_features=out_features
    )
