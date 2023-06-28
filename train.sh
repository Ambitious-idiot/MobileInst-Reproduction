#!/bin/bash
train_file=$(dirname $0)/tools/train_net.py
config_file=$(dirname $0)/configs/mobileinst_topformer_base_coco.yaml
dataset=voc2012_cocostyle
train_json=/home/yuehaosong/datasets/VOC2012/VOCdevkit/VOC2012/voc_2012_val_cocostyle_split_train.json
val_json=/home/yuehaosong/datasets/VOC2012/VOCdevkit/VOC2012/voc_2012_val_cocostyle_split_val.json
image_root=/home/yuehaosong/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages
python $train_file --config-file $config_file --num-gpus 4 SOLVER.AMP.ENABLED True
