import random
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import argparse


# ArgumentParser
def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--root', type=str, default='./dataset/voc2012/VOCdevkit/VOC2012', help='Root directory of dataset')
    parser.add_argument('--n_cls', type=int, default=20, help='Number of classes')
    # MobileInst
    parser.add_argument('--channels', type=tuple, default=(128, 128, 128), help='Output channels of backbone')
    parser.add_argument('--dim', type=int, default=128, help='Kernel dim')
    parser.add_argument('--key_dim', type=int, default=16, help='Key dim of attention')
    parser.add_argument('--num_heads', type=int, default=8, help='Head of attention')
    parser.add_argument('--attn_ratios', type=int, default=2, help='Attention ratios')
    parser.add_argument('--mlp_ratios', type=int, default=2, help='MLP ratios in attention')
    # training config
    parser.add_argument('--max_iter', type=int, default=14400, help='Max iterations')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--resume', type=bool, default=False, help='Resume with args.checkpoint')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints', help='Root of checkpoints')
    parser.add_argument('--checkpoint', type=str, default='mobileinst2250.pth', help='Path to checkpoint to load')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00005, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay of regularation')
    parser.add_argument('--steps', type=tuple, default=(11200, 13200), help='Steps that lr drops')
    parser.add_argument('--backbone_multiplier', type=float, default=0.5, help='')
    # loss
    parser.add_argument('--ce_weight', type=float, default=2, help='Weight of class error')
    parser.add_argument('--mask_weight', type=float, default=2, help='Weight of mask loss')
    parser.add_argument('--dice_weight', type=float, default=2, help='Weight of dice loss')
    parser.add_argument('--obj_weight', type=float, default=1, help='Weight of objectness')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight of mask score in matcher')
    parser.add_argument('--beta', type=float, default=1., help='Weight of class score in matcher')
    # Validation Config
    parser.add_argument('--cls_thres', type=float, default=0.2, help='Score threshold for inference')
    parser.add_argument('--mask_thres', type=float, default=0.3, help='Mask threshold for inference')
    parser.add_argument('--out_root', type=str, default='./out', help='Root of output results')
    return parser.parse_args()


# RandomSeed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Visualization
def tensor2masks(tensors: torch.TensorType):
    assert tensors.dim() in [2, 3]
    if tensors.dim() == 2:
        tensors = tensors.unsqueeze(0)
    tensors = (tensors > 0.5) * 255
    imgs = [Image.fromarray(tensor.byte().cpu().numpy()) \
            for tensor in tensors]
    return imgs


def batch2images(tensors: torch.TensorType):
    assert tensors.dim() in [3, 4]
    if tensors.dim() == 3:
        tensors = tensors.unsqueeze(0)
    tensors = (tensors - tensors.min()) / (tensors.max()-tensors.min()) * 255
    imgs = [F.to_pil_image(tensor) for tensor in tensors.byte()]
    return imgs


def gen_color_table(n_instances):
    assert n_instances < 37
    color_wheel = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 1., 0.],
        [1., 0., 1.],
        [0., 1., 1.],
    ])
    bg_color = np.zeros(3)
    colors = np.zeros((n_instances, 3))
    for i in range(n_instances):
        base_color = color_wheel[i % 6]
        t = (i // 6) / 6
        colors[i] = bg_color * t + base_color * (1-t)
    return torch.from_numpy(colors).float()


def fuse_masks(masks: torch.TensorType, to_image: bool = False):
    assert masks.dim() in [2, 3]
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)

    n_instances = len(masks)
    color_tabel = gen_color_table(n_instances)

    mask = torch.zeros(*masks.shape[-2:], 3)
    for m, c in zip(masks, color_tabel):
        mask[m > 0.5] = c
    mask = mask.permute(2, 0, 1)
    if to_image:
        mask = F.to_pil_image((mask * 255).byte())
    return mask
