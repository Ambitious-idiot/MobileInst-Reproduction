from torch import nn
import torch.nn.functional as F
from detectron2.utils.registry import Registry
from fvcore.nn.weight_init import c2_xavier_fill
from detectron2.layers import NaiveSyncBatchNorm, FrozenBatchNorm2d
from typing import Literal


MOBILEINST_ENCODER_REGISTRY = Registry("MOBILEINST_ENCODER")
MOBILEINST_ENCODER_REGISTRY.__doc__ = "Registry for MobileInst Encoder"


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


class SemanticEnhancer(nn.Module):
    def __init__(self,
                 in_local: int,
                 in_glob: int,
                 out_feat: int):
        super(SemanticEnhancer, self).__init__()
        self.in_local = in_local
        self.in_glob = in_glob
        self.out_feat = out_feat

        self.lateral_l = Conv2d_BN(in_local, out_feat)
        self.lateral_g = Conv2d_BN(in_glob, out_feat)
        self.out_layer = Conv2d_BN(out_feat, out_feat)

    def forward(self, x_l, x_g):
        x_l = F.relu_(self.lateral_l(x_l))
        x_g = F.relu_(self.lateral_g(x_g))
        features = x_l * x_g + x_g
        features = F.relu_(self.out_layer(features))
        return features


@ MOBILEINST_ENCODER_REGISTRY.register()
class SEMaskEncoder(nn.Module):
    def __init__(self, cfg, input_shape):
        super(SEMaskEncoder, self).__init__()
        self.num_channels = cfg.MODEL.MOBILEINST.ENCODER.NUM_CHANNELS
        self.in_features = cfg.MODEL.MOBILEINST.ENCODER.IN_FEATURES
        norm = cfg.MODEL.MOBILEINST.ENCODER.NORM
        self.in_channels = [input_shape[f].channels for f in self.in_features]
        self.n_local = len(self.in_channels) - 1

        self.injections = nn.ModuleList()
        for channel in self.in_channels:
            self.injections.append(nn.Sequential(
                Conv2d_BN(channel, self.num_channels, norm=norm), nn.ReLU(inplace=True)))
        self.se1 = SemanticEnhancer(self.num_channels, self.num_channels, self.num_channels)
        self.conv1 = nn.ModuleList()
        for _ in range(self.n_local):
            self.conv1.append(nn.Sequential(
                Conv2d_BN(self.num_channels, self.num_channels, 3, 1, 1, norm=norm), nn.ReLU(inplace=True)))
        self.conv2 = nn.ModuleList()
        for _ in range(self.n_local):
            self.conv2.append(nn.Sequential(
                Conv2d_BN(self.num_channels, self.num_channels, 3, 1, 1, norm=norm), nn.ReLU(inplace=True)))
        self.se2 = SemanticEnhancer(self.num_channels, self.num_channels, self.num_channels)

    def forward(self, features):
        features = [features[f] for f in self.in_features]
        x_l = [self.injections[i](features[i]) for i in range(self.n_local)]
        x_g = self.injections[-1](features[-1])
        x_g = F.interpolate(
            x_g, size=x_l[-1].shape[-2:], mode='bilinear', align_corners=False)
        stage1 = [self.se1(x_l[-1], x_g)]
        for i in range(self.n_local-2, -1, -1):
            up_feature = F.interpolate(
                stage1[-1], size=x_l[i].shape[-2:], mode='bilinear', align_corners=False)
            stage1.append(up_feature+x_l[i])
        stage1 = stage1[::-1]
        stage1 = [self.conv1[i](stage1[i]) for i in range(self.n_local)]
        stage2 = [stage1[0]]
        for i in range(1, self.n_local):
            down_feature = F.interpolate(
                stage2[-1], size=stage1[i].shape[-2:], mode='bilinear', align_corners=False)
            stage2.append(down_feature+stage1[i])
        stage2 = [self.conv2[i](stage2[i]) for i in range(self.n_local)]
        out = self.se2(stage2[-1], x_g)
        for i in range(self.n_local-2, -1, -1):
            out = F.interpolate(
                out, size=stage2[i].shape[-2:], mode='bilinear', align_corners=False)
            out = out + stage2[i]
        return {
            "x_local": out,
            "x_global": x_g
        }

def build_mobileinst_encoder(cfg, input_shape):
    name = cfg.MODEL.MOBILEINST.ENCODER.NAME
    return MOBILEINST_ENCODER_REGISTRY.get(name)(cfg, input_shape)
