#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/
NVCC=/usr/local/cuda-9.0/bin/nvcc
cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
