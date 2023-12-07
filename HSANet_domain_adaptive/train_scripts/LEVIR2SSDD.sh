#!/bin/sh
CUDA_VISIBLE_DEVICES=$0 python trainval_val.py --cuda --net res101 --dataset LEVIR --dataset_t SSDD 