#!/bin/bash
CUDA_VER=9.2
CUDNN_VER=7
NCCL_VER=2.3.7-1
PYTORCH_VER=1.0rc1
CONDA_VER=5.2.0
PY_VER=3.6

module load gcc/6.4.0
module load magma/2.3.0 cuda/9.2.148
#install pre-built anaconda
wget https://repo.anaconda.com/archive/Anaconda3-$CONDA_VER-Linux-ppc64le.sh -O anaconda3.sh  
bash ./anaconda3.sh -b -p anaconda3 
export PATH=$(pwd)/anaconda3/bin:$PATH
pip install --upgrade pip

#workaround for nccl2
git clone https://github.com/NVIDIA/nccl
cd nccl
git checkout v$NCCL_VER
make -j src.build CUDA_HOME=$CUDA_DIR
make pkg.txz.build CUDA_HOME=$CUDA_DIR
tar -xf  build/pkg/txz/* -C ..
cd ..
ln -s nccl_$NCLL_VER* nccl2

#Download your own cudnn binary
#wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/v7.2.1/prod/9.2_20180806/cudnn-9.2-linux-ppc64le-v7.2.1.38.tgz
#mkdir cudnn-9.2-linux-ppc64le-v7.2.1.38 && tar -xf cudnn-9.2-linux-ppc64le-v7.2.1.38.tgz -C cudnn-9.2-linux-ppc64le-v7.2.1.38
#ln -s cudnn-9.2-linux-ppc64le-v7.2.1.38/cuda/targets/ppc64le-linux cudnn
#ln -s lib cudnn/lib64
#CUDNN_DIR=$(pwd)/cudnn
#or use the one from cuda module
CUDNN_DIR=$CUDA_DIR


#setup env var
export LD_LIBRARY_PATH=$(pwd)/anaconda3/lib:$(pwd)/nccl2/lib:$CUDNN_DIR/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
conda install -y numpy pyyaml setuptools cmake cffi

#build pytorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch/
git checkout v$PYTORCH_VER

CC=mpicc CXX=mpic++ USE_GLOO_IBVERBS=1 CUDA_HOME=$CUDA_DIR CUDNN_INCLUDE_DIR=$CUDNN_DIR/include CUDNN_LIB_DIR=$CUDNN_DIR/lib NCCL_ROOT_DIR=$(pwd)/nccl2  python setup.py bdist_wheel 2>&1 | tee log.build

cp dist/*.whl ../wheels
#pip install dist/torch*.whl

#HOROVOD_WITH_PYTORCH=1  HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=./build-tensorflow/nccl2 HOROVOD_CUDA_HOME=/sw/summit/cuda/9.2.88  pip --no-cache-dir install horovod
#pip install torchvision tensorboardX tqdm




