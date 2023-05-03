#!/bin/bash
root=$(dirname $0)/../../datasets/VOC2012/VOCdevkit/VOC2012
checkpoint_root=$(dirname $0)/../checkpoints
out_root=$(dirname $0)/../out
checkpoint=$(ls $checkpoint_root | sort -r | head -n 1)
python -u $(dirname $0)/../val.py --root $root --checkpoint_root $checkpoint_root --checkpoint $checkpoint --out_root $out_root
