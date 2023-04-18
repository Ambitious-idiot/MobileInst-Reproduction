import torch
from torch import nn
import torch.nn.functional as F
import math
from detectron2.layers import NaiveSyncBatchNorm, FrozenBatchNorm2d


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class SemanticEnhancer(nn.Module):
    def __init__(self):
        super(SemanticEnhancer, self).__init__()

    def forward(self, x_l, x_g):
        return x_l * x_g + x_g


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, norm='BN'):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        if norm == "FrozenBN":
            bn = FrozenBatchNorm2d(b)
        elif norm == "SyncBN":
            bn = NaiveSyncBatchNorm(b)
        else:
            bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class SEMaskDecoder(nn.Module):
    def __init__(self, channels, dim, activation=nn.ReLU, norm='bn'):
        super(SEMaskDecoder, self).__init__()
        self.n_local = len(channels) - 1
        self.injections = nn.ModuleList()
        for channel in channels:
            self.injections.append(nn.Sequential(Conv2d_BN(channel, dim, norm=norm), activation()))
        self.se1 = SemanticEnhancer()
        self.conv1 = nn.ModuleList()
        for _ in range(self.n_local):
            self.conv1.append(nn.Sequential(Conv2d_BN(dim, dim, 3, 1, 1, norm=norm), activation()))
        self.conv2 = nn.ModuleList()
        for _ in range(self.n_local):
            self.conv2.append(nn.Sequential(Conv2d_BN(dim, dim, 3, 1, 1, norm=norm), activation()))
        self.se2 = SemanticEnhancer()

    def forward(self, x_l, x_g):
        x_l = [self.injections[i](x_l[i]) for i in range(self.n_local)]
        x_g = self.injections[-1](x_g)
        stage1 = [self.se1(x_l[-1], x_g)]
        for i in range(self.n_local-2, -1, -1):
            stage1.append(stage1[-1]+x_l[i])
        stage1 = stage1[::-1]
        stage1 = [self.conv1[i](stage1[i]) for i in range(self.n_local)]
        stage2 = [stage1[0]]
        for i in range(1, len(stage1)):
            stage2.append(stage2[-1]+stage1[i])
        stage2 = [self.conv2[i](stage2[i]) for i in range(self.n_local)]
        out = self.se2(stage2[-1], x_g)
        for i in range(len(stage2)-1):
            out = out + stage2[i]
        return out


class BaseAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4,
                 activation=nn.ReLU, norm='bn'):
        super(BaseAttention, self).__init__()
        self.dim = dim
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
        # Q: B*NH*C*KD K: B*NH*KD*HW V: B*NH*HW*D Out: B*C*C
        attn = torch.matmul(q, k)
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, v)
        x = x.permute(0, 1, 3, 2).reshape(B, self.dh, self.dim).permute(0, 2, 1).reshape(-1, self.dh)
        return self.proj(x).reshape(-1, self.dim, self.dim)

    def attention(self, q, k, v, B):
        # Q: B*C*C K: B*C*H*W V: B*C*H*W Out: B*C*C
        Q = q.reshape(-1, self.dim)
        Q = self.to_q(Q).reshape(B, self.dim, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        K = self.to_k(k).reshape(B, self.num_heads, self.key_dim, -1)
        V = self.to_v(v).reshape(B, self.num_heads, self.d, -1).permute(0, 1, 3, 2)
        return self._attention(Q, K, V, B) + q


class FixedQAttention(BaseAttention):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4,
                 activation=nn.ReLU, norm='bn'):
        super(FixedQAttention, self).__init__(dim, key_dim, num_heads, attn_ratio, activation, norm)
        self.q = nn.Parameter(torch.randn(1, dim, dim))

    def forward(self, x):
        B, *_ = get_shape(x)
        q = self.q.repeat(B, 1, 1)
        return self.attention(q, x, x, B)


class CrossAttention(BaseAttention):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4,
                 activation=nn.ReLU, norm='bn'):
        super(CrossAttention, self).__init__(dim, key_dim, num_heads, attn_ratio, activation, norm)

    def forward(self, x, y):
        B, *_ = get_shape(x)
        return self.attention(y, x, x, B)


class SelfAttention(BaseAttention):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4,
                 activation=nn.ReLU, norm='bn'):
        super(SelfAttention, self).__init__(dim, key_dim, num_heads, attn_ratio, activation, norm)
        self.to_k = nn.Sequential(nn.Linear(dim, self.nh_kd), nn.BatchNorm1d(self.nh_kd))
        self.to_v = nn.Sequential(nn.Linear(dim, self.dh), nn.BatchNorm1d(self.dh))

    def attention(self, q, k, v, B):
        # Q: B*C*C K: B*C*C V: B*C*C Out: B*C*C
        Q = q.reshape(-1, self.dim)
        Q = self.to_q(Q).reshape(B, self.dim, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        K = k.reshape(-1, self.dim)
        K = self.to_k(K).reshape(B, self.dim, self.num_heads, self.key_dim).permute(0, 2, 3, 1)
        V = v.reshape(-1, self.dim)
        V = self.to_v(V).reshape(B, self.dim, self.num_heads, self.d).permute(0, 2, 1, 3)
        return self._attention(Q, K, V, B) + q

    def forward(self, x):
        B, *_ = get_shape(x)
        return self.attention(x, x, x, B)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
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


class PredictionHead(nn.Module):
    def __init__(self, dim, n_cls):
        super(PredictionHead, self).__init__()
        self.dim = dim
        self.n_cls = n_cls
        self.cls = nn.Linear(dim, n_cls)
        self.objectness = nn.Linear(dim, 1)
        self.kernel = nn.Linear(dim, dim)

    def forward(self, x):
        pred_logits = self.cls(x)
        pred_kernel = self.kernel(x)
        pred_scores = self.objectness(x)
        return pred_logits, pred_kernel, pred_scores


class DualInstanceDecoder(nn.Module):
    def __init__(self, dim, key_dim, num_heads, n_cls, mlp_ratio=4., attn_ratio=4,
                 activation=nn.ReLU, norm='bn'):
        super(DualInstanceDecoder, self).__init__()
        self.pe = PositionEmbeddingSine(dim//2, normalize=True)
        self.g_d1 = FixedQAttention(dim, key_dim, num_heads, attn_ratio, activation, norm)
        self.g_d2 = SelfAttention(dim, key_dim, num_heads, attn_ratio, activation, norm)
        self.l_d1 = CrossAttention(dim, key_dim, num_heads, attn_ratio, activation, norm)
        self.l_d2 = SelfAttention(dim, key_dim, num_heads, attn_ratio, activation, norm)
        self.mlp = MLP(dim, int(dim*mlp_ratio), act_layer=activation)
        self.head = PredictionHead(dim, n_cls)

    def forward(self, x_l, x_g):
        x_l = x_l + self.pe(x_l)
        x_g = x_g + self.pe(x_g)
        q = self.g_d1(x_g)
        q = self.g_d2(q)
        q = self.l_d1(x_l, q)
        q = self.l_d2(q)
        q = self.mlp(q)
        pred_logits, pred_kernel, pred_scores = self.head(q)
        return pred_logits, pred_kernel, pred_scores


class MobileInst(nn.Module):
    def __init__(self, backbone, channels, dim, key_dim, num_heads, n_cls, mlp_ratio=4., attn_ratio=4,
                 activation=nn.ReLU, norm='bn'):
        super(MobileInst, self).__init__()
        self.backbone = backbone
        self.se_decoder = SEMaskDecoder(channels, dim, activation, norm)
        self.dual_decoder = DualInstanceDecoder(dim, key_dim, num_heads, n_cls, mlp_ratio, attn_ratio, activation, norm)

    def forward(self, x):
        _, __, H, W = get_shape(x)
        features = self.backbone(x)
        x_g = features[-1]
        x_l = features[:-1]
        x_l = [F.interpolate(x, size=(H//8, W//8), mode='bilinear', align_corners=False) for x in x_l]
        x_g = F.interpolate(x_g, size=(H//8, W//8), mode='bilinear', align_corners=False)
        x_mask = self.se_decoder(x_l, x_g)
        x_l = F.interpolate(x_mask, size=(H//64, W//64), mode='bilinear', align_corners=False)
        x_g = F.interpolate(x_g, size=(H//64, W//64), mode='bilinear', align_corners=False)
        pred_logits, pred_kernels, pred_scores = self.dual_decoder(x_l, x_g)
        mask_shape = get_shape(x_mask)
        pred_masks = torch.bmm(pred_kernels, x_mask.view(mask_shape[0], mask_shape[1], -1)).view(*mask_shape)
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear', align_corners=False)
        return {
            'pred_logits': pred_logits,
            'pred_scores': pred_scores,
            'pred_masks': pred_masks
        }
