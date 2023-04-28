import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from fvcore.nn import sigmoid_focal_loss_jit


def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_score(inputs, targets):
    targets = targets.float()
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (
        inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()


class MobileInstCriterion(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py
    def __init__(self, weights, n_cls, matcher, device):
        super().__init__()
        self.matcher = matcher
        self.losses = ('labels', 'masks')
        self.weight_dict = self.get_weight_dict(weights)
        self.num_classes = n_cls
        self.device = device

    def get_weight_dict(self, weights):
        losses = ("loss_ce", "loss_mask", "loss_dice", "loss_objectness")
        return dict(zip(losses, weights))

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_instances):
        src_logits = outputs['pred_logits']
        labels = torch.zeros_like(src_logits)
        src_idx = self._get_src_permutation_idx(indices)
        src_logits = src_logits[src_idx]
        tgt_idx = self._get_tgt_permutation_idx(indices)
        targets = torch.cat(targets['labels']).to(self.device)
        labels[tgt_idx] = targets.float()
        labels = labels.flatten(0, 1).to(self.device)
        class_loss = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        ) / num_instances
        losses = {'loss_ce': class_loss}
        return losses

    def loss_masks_with_iou_objectness(self, outputs, targets, indices, num_instances):
        indices = [(i[0][:len(i[1])], i[1]) for i in indices]
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        tgt_masks = targets['masks']
        tgt_masks = [mask[index[1]] for mask, index in zip(tgt_masks, indices)]
        tgt_masks = torch.cat(tgt_masks)
        tgt_masks = F.interpolate(tgt_masks.unsqueeze(1), size=src_masks.shape[-2:], mode='nearest').squeeze(1)
        src_masks = src_masks[src_idx].flatten(1)
        tgt_masks = tgt_masks.float().flatten(1).to(self.device)
        src_iou_scores = outputs["pred_scores"][src_idx].flatten(0)
        with torch.no_grad():
            tgt_iou_scores = compute_mask_iou(src_masks, tgt_masks).flatten(0)
        losses = {
            "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            "loss_dice": dice_loss(src_masks, tgt_masks) / num_instances,
            "loss_mask": F.binary_cross_entropy_with_logits(src_masks, tgt_masks, reduction='mean')
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_instances):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks_with_iou_objectness,
        }
        return loss_map[loss](outputs, targets, indices, num_instances)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_instances = sum(len(tl) for tl in targets['labels'])
        num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=self.device)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_instances))

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses


class MobileInstMatcher:
    def __init__(self, alpha, beta, num_cands, device):
        self.alpha = alpha
        self.beta = beta
        self.mask_score = dice_score
        self.num_cands = num_cands
        self.device = device

    def __call__(self, outputs, targets):
        with torch.no_grad():
            pred_masks = outputs['pred_masks']
            batch_size = len(pred_masks)
            pred_logits = outputs['pred_logits'].sigmoid()

            indices = []

            for i in range(batch_size):
                target_labels = targets['labels'][i].to(self.device)
                if target_labels.shape[0] == 0:
                    indices.append((torch.as_tensor([]),
                                    torch.as_tensor([])))
                    continue

                tgt_masks = targets['masks'][i].to(self.device)
                pred_logit = pred_logits[i]
                out_masks = pred_masks[i]
                tgt_masks = F.interpolate(
                    tgt_masks.unsqueeze(1),size=out_masks.shape[-2:],
                    mode='nearest').squeeze(1)

                # compute dice score and classification score
                tgt_masks = tgt_masks.flatten(1)
                out_masks = out_masks.flatten(1)
                mask_score = self.mask_score(out_masks, tgt_masks)
                matching_prob = torch.matmul(pred_logit, target_labels.T.float())
                c = (mask_score ** self.alpha) * (matching_prob ** self.beta)
                # hungarian matching
                src, tgt = linear_sum_assignment(c.cpu(), maximize=True)
                src = np.hstack([src, np.setdiff1d(np.arange(self.num_cands), src, True)])
                indices.append((src, tgt))
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
