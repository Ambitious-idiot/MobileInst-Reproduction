import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from pycocotools.coco import COCO
from PIL import Image
from functools import partial
import random
import os


__all__ = [
    'build_train_dataloader',
    'build_val_dataloader'
]


class CocoInstanceSegmentationDataset(Dataset):
    def __init__(self, root_dir, split, n_cls, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.n_cls = n_cls
        self.transform = transform
        self.img_transform = transforms.ToTensor()

        self.coco = COCO(os.path.join(root_dir, "voc_2012_val_cocostyle_split_{}.json".format(self.split)))

        self.image_ids = self.coco.getImgIds()

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        target = {}
        # 加载图像并进行数据增强
        image = self.coco.loadImgs(img_id)[0]
        img_path = "JPEGImages/{}".format(image['file_name'])
        img = Image.open(os.path.join(self.root_dir, img_path)).convert("RGB")
        size = img.size
        img = self.img_transform(img)
        # 创建标签图像和实例掩码
        labels = torch.zeros(((len(annotations), self.n_cls)), dtype=torch.int64)
        masks = torch.zeros((len(annotations), size[1], size[0]), dtype=torch.uint8)
        for i, ann in enumerate(annotations):
            labels[i, ann['category_id']-1] = 1
            mask = self.coco.annToMask(ann)
            masks[i, :, :] = torch.from_numpy(mask)

        if self.transform is not None:
            img, masks = self.transform(img, masks)

        target["image"] = img
        target["labels"] = labels
        target["masks"] = masks

        return target

    def __len__(self):
        return len(self.image_ids)


def train_collate(batch):
    max_h = max([sample['image'].shape[-2] for sample in batch])
    max_w = max([sample['image'].shape[-1] for sample in batch])
    images, masks, labels = [], [], []
    for sample in batch:
        pad_h = max_h - sample['image'].shape[-2]
        pad_w = max_w - sample['image'].shape[-1]
        padded = F.pad(sample['image'],
                       (pad_w // 2, pad_w - pad_w // 2,
                        pad_h // 2, pad_h - pad_h // 2),
                       mode='constant', value=0)
        images.append(padded)
        padded = F.pad(sample['masks'],
                       (pad_w // 2, pad_w - pad_w // 2,
                        pad_h // 2, pad_h - pad_h // 2),
                       mode='constant', value=0)
        masks.append(padded)
        labels.append(sample['labels'])
    images = torch.stack(images, dim=0)
    images = transforms.functional.normalize(
        images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
    targets = {}
    targets['image'] = images
    targets['masks'] = masks
    targets['labels'] = labels
    return targets


def test_collate(batch):
    for i in range(len(batch)):
        batch[i]['image'] = transforms.functional.normalize(
            batch[i]['image'], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
    return batch


def train_transform(img, masks, p_h=0.5, p_v=0.5, long_edge=864):
    # RandomFlip
    if torch.rand(1) < p_h:
        img = transforms.functional.hflip(img)
        masks = transforms.functional.hflip(masks)
    if torch.rand(1) < p_v:
        img = transforms.functional.vflip(img)
        masks = transforms.functional.vflip(masks)
    # ScaleJitter
    short_edge = random.choice([416, 448, 480, 512, 544, 576, 608, 640])
    h, w = img.shape[-2:]
    if h > w:
        h = min(h * short_edge // w, long_edge)
        w = short_edge
    else:
        w = min(w * short_edge // h, long_edge)
        h = short_edge
    img = transforms.functional.resize(img, (h, w))
    masks = transforms.functional.resize(masks, (h, w), transforms.functional.InterpolationMode.NEAREST)
    return img, masks


def build_train_dataloader(root_dir, n_cls, batch_size, **kwargs):
    dataset = CocoInstanceSegmentationDataset(root_dir, 'train', n_cls, partial(train_transform, **kwargs))
    return DataLoader(dataset, batch_size, True, collate_fn=train_collate)


def build_val_dataloader(root_dir, n_cls):
    dataset = CocoInstanceSegmentationDataset(root_dir, 'val', n_cls)
    return DataLoader(dataset, 1, False, collate_fn=test_collate)
