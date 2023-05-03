import torch
from torchvision.transforms.functional import normalize
import numpy as np
import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from typing import Dict
import os
import json
from .dataloader import CocoInstanceSegmentationDataset


class Evaluator:
    def __init__(self, args, model, device):
        self.dataset = CocoInstanceSegmentationDataset(args.root, 'val', 20)
        self.image_ids  = self.dataset.image_ids
        self.coco = self.dataset.coco
        self.model = model.eval().to(device)
        self.cls_thres = args.cls_thres
        self.mask_thres = args.mask_thres
        self.device = device
        self.out_root = args.out_root
        os.makedirs(self.out_root, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def inference(self, idx: int, to_coco: bool = False):
        assert idx < len(self)
        data = self.dataset[idx]
        image = normalize(
            data['image'],
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
            inplace=True).to(self.device)

        out = self.model(image.unsqueeze(0))
        pred_scores = out['pred_logits'].sigmoid().squeeze(0)
        pred_masks = out['pred_masks'].sigmoid().squeeze(0)
        pred_objectness = out["pred_scores"].sigmoid().squeeze(0)
        pred_scores = torch.sqrt(pred_scores * pred_objectness)

        scores, labels = pred_scores.max(dim=-1)
        keep = scores > self.cls_thres
        scores = scores[keep]
        labels = labels[keep]
        masks = pred_masks[keep]

        result = {}
        if scores.size(0) == 0:
            result['scores'] = scores
            result['pred_classes'] = labels
            result['num_instance'] = 0
            return result if not to_coco else result2coco(result, self.image_ids[idx])

        masks_pred = masks > self.mask_thres
        # rescoring mask using maskness
        scores = rescoring_mask(scores, masks_pred, masks)
        result['pred_masks'] = masks_pred
        result['scores'] = scores
        result['pred_classes'] = labels + 1
        result['num_instance'] = len(scores)
        return result if not to_coco else result2coco(result, self.image_ids[idx])

    def gen_result_json(self):
        results = []
        for i in range(len(self)):
            results.extend(self.inference(i, True))
        res_file= os.path.join(self.out_root, 'segment_coco_results.json')
        with open(res_file, 'w') as f:
            json.dump(results, f)
        return res_file

    def evaluate(self, res_file):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        info_str = []
        stats_names = ['AP', 'Ap .5', 'AP .75','AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        return info_str


def rescoring_mask(scores, masks_pred, masks):
    mask_pred_ = masks_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))


def result2coco(result: Dict, img_id: int):
    num_instance = result['num_instance']
    if num_instance == 0:
        return []

    scores = result['scores'].tolist()
    classes = result['pred_classes'].tolist()

    rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in result['pred_masks']
        ]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")

    return [
        {
            'image_id': img_id,
            "category_id": classes[i],
            "score": scores[i],
            "segmentation": rles[i]
        } for i in range(num_instance)
    ]
