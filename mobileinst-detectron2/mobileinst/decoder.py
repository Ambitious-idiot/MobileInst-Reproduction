# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional
from detectron2.utils.registry import Registry
from .position_encoding import PositionEmbeddingSine


MOBILEINST_DECODER_REGISTRY = Registry("MOBILEINST_DECODER")
MOBILEINST_DECODER_REGISTRY.__doc__ = "Registry for MobileInst Decoder"


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DualInstanceDecoder(nn.Module):
    def __init__(self, cfg):
        super(DualInstanceDecoder, self).__init__()
        in_channels = cfg.MODEL.MOBILEINST.DECODER.IN_CHANNELS
        num_classes = cfg.MODEL.MOBILEINST.DECODER.NUM_CLASSES
        mask_dim = cfg.MODEL.MOBILEINST.DECODER.MASK_DIM
        hidden_dim = cfg.MODEL.MOBILEINST.DECODER.HIDDEN_DIM
        num_queries = cfg.MODEL.MOBILEINST.DECODER.NUM_QUERIES
        num_heads = cfg.MODEL.MOBILEINST.DECODER.NUM_HEADS
        dim_feedforward = cfg.MODEL.MOBILEINST.DECODER.DIM_FEEDFORWARD
        pre_norm = cfg.MODEL.MOBILEINST.DECODER.PRE_NORM

        self.pe = PositionEmbeddingSine(in_channels//2, normalize=True)

        self.g_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.l_proj = nn.Conv2d(in_channels, hidden_dim, 1)

        self.g_d1 = CrossAttentionLayer(hidden_dim, num_heads, normalize_before=pre_norm)
        self.g_d2 = SelfAttentionLayer(hidden_dim, num_heads, normalize_before=pre_norm)
        self.l_d1 = CrossAttentionLayer(hidden_dim, num_heads, normalize_before=pre_norm)
        self.l_d2 = SelfAttentionLayer(hidden_dim, num_heads, normalize_before=pre_norm)
        self.mlp = FFNLayer(hidden_dim, dim_feedforward, normalize_before=pre_norm)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.mask_kernel = nn.Linear(hidden_dim, mask_dim)

    def forward(self, x_l, x_g):
        pos_l = self.pe(x_l).flatten(2).permute(2, 0, 1)
        pos_g = self.pe(x_g).flatten(2).permute(2, 0, 1)
        x_l = self.l_proj(x_l).flatten(2).permute(2, 0, 1)
        x_g = self.g_proj(x_g).flatten(2).permute(2, 0, 1)

        B = x_g.shape[1]

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        query = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)

        query = self.g_d1(query, x_g, pos=pos_g, query_pos=query_embed)
        query = self.g_d2(query, query_pos=query_embed)
        query = self.l_d1(query, x_l, pos=pos_l, query_pos=query_embed)
        query = self.l_d2(query, query_pos=query_embed)
        query = self.mlp(query)
        query = self.decoder_norm(query)
        query = query.transpose(0, 1)
        pred_logits = self.cls_score(query)
        pred_kernel = self.mask_kernel(query)
        pred_scores = self.objectness(query)
        return pred_logits, pred_kernel, pred_scores


@ MOBILEINST_DECODER_REGISTRY.register()
class MobileInstDecoder(nn.Module):
    def __init__(self, cfg):
        super(MobileInstDecoder, self).__init__()
        self.scale_factor = cfg.MODEL.MOBILEINST.DECODER.SCALE_FACTOR
        in_channels = cfg.MODEL.MOBILEINST.DECODER.IN_CHANNELS
        mask_dim = cfg.MODEL.MOBILEINST.DECODER.MASK_DIM

        self.inst_branch = DualInstanceDecoder(cfg)
        self.mask_branch = nn.Conv2d(in_channels, mask_dim, 1)

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
