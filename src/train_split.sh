#!/bin/bash
for m in {0..49}
do
   cmd="CUDA_VISIBLE_DEVICES=0 python train_feat.py --exp_id=$m"
   echo "${cmd}"
   eval ${cmd}
done

