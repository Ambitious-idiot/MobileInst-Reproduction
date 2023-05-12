import torch
from torch import nn
import torch.nn.functional as F
from detectron2.utils.registry import Registry
from .modules import (CrossAttention, SelfAttention, FixedQAttention,
                      MLP, PositionEmbeddingSine)


MOBILEINST_DECODER_REGISTRY = Registry("MOBILEINST_DECODER")
MOBILEINST_DECODER_REGISTRY.__doc__ = "registry for MobileInst decoder"


class DualInstanceDecoder(nn.Module):
    def __init__(self, cfg, in_channel: int, activation=nn.GELU):
        super(DualInstanceDecoder, self).__init__()
        kernel_dim = cfg.MODEL.MOBILEINST.DECODER.KERNEL_DIM
        num_kernels = cfg.MODEL.MOBILEINST.DECODER.NUM_MASKS
        key_dim = cfg.MODEL.MOBILEINST.DECODER.KEY_DIM
        num_heads = cfg.MODEL.MOBILEINST.DECODER.NUM_HEADS
        mlp_ratio = cfg.MODEL.MOBILEINST.DECODER.MLP_RATIO
        attn_ratio = cfg.MODEL.MOBILEINST.DECODER.ATTN_RATIO
        norm = cfg.MODEL.MOBILEINST.NORM
        n_cls = cfg.MODEL.MOBILEINST.DECODER.NUM_CLASSES
        self.pe = PositionEmbeddingSine(in_channel//2, normalize=True)
        self.g_d1 = FixedQAttention(in_channel, num_kernels, key_dim, num_heads, attn_ratio, activation, norm)
        self.g_d2 = SelfAttention(in_channel, num_kernels, key_dim, num_heads, attn_ratio, activation, norm)
        self.l_d1 = CrossAttention(in_channel, num_kernels, key_dim, num_heads, attn_ratio, activation, norm)
        self.l_d2 = SelfAttention(in_channel, num_kernels, key_dim, num_heads, attn_ratio, activation, norm)
        self.mlp = MLP(in_channel, int(in_channel*mlp_ratio))
        self.cls_score = nn.Linear(in_channel, n_cls)
        self.objectness = nn.Linear(in_channel, 1)
        self.mask_kernel = nn.Linear(in_channel, kernel_dim)

    def forward(self, x_l, x_g):
        x_l = x_l + self.pe(x_l)
        x_g = x_g + self.pe(x_g)
        q = self.g_d1(x_g)
        q = self.g_d2(q)
        q = self.l_d1(x_l, q)
        q = self.l_d2(q)
        q = self.mlp(q)
        pred_logits = self.cls_score(q)
        pred_kernel = self.mask_kernel(q)
        pred_scores = self.objectness(q)
        return pred_logits, pred_kernel, pred_scores


@ MOBILEINST_DECODER_REGISTRY.register()
class MobileInstDecoder(nn.Module):
    def __init__(self, cfg):
        super(MobileInstDecoder, self).__init__()
        in_channel = cfg.MODEL.MOBILEINST.ENCODER.NUM_CHANNELS
        self.scale_factor = cfg.MODEL.MOBILEINST.DECODER.SCALE_FACTOR
        kernel_dim = cfg.MODEL.MOBILEINST.DECODER.KERNEL_DIM

        self.inst_branch = DualInstanceDecoder(cfg, in_channel)
        self.mask_branch = nn.Conv2d(in_channel, kernel_dim, 1)

    def forward(self, features):
        x_l, x_g = features['x_local'], features['x_global']
        mask_features = self.mask_branch(x_l)
        x_l = F.adaptive_max_pool2d(x_l, output_size=x_g.shape[-2:])
        pred_logits, pred_kernel, pred_scores = self.inst_branch(x_l, x_g)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, mask_features.view(
            B, C, H * W)).view(B, N, H, W)

        pred_masks = F.interpolate(
            pred_masks, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)

        return {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
        }


def build_mobileinst_decoder(cfg):
    name = cfg.MODEL.MOBILEINST.DECODER.NAME
    return MOBILEINST_DECODER_REGISTRY.get(name)(cfg)
