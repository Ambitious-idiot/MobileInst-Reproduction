#!/bin/bash
checkpoint_root=$(dirname $0)/../checkpoints
checkpoint=$(ls $checkpoint_root | sort -r | head -n 1)
python -u  $(dirname $0)/../train.py --resume True --checkpoint_root $checkpoint_root --checkpoint $checkpoint
