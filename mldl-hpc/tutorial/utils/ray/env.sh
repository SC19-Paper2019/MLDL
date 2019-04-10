#!/bin/bash
module load gcc magma
module load cuda/9.2.148
PREFIX=/gpfs/alpine/world-shared/stf011/junqi/Ray
export PYTHONPATH=$PREFIX/ray/python
export LD_LIBRARY_PATH=$PREFIX/nccl/lib:$PREFIX/cudnn/lib:$LD_LIBRARY_PATH
export PATH=$PREFIX/anaconda3/bin:$PATH
