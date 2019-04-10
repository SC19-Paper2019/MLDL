#!/bin/bash

CUDA_VER=9.2
CUDNN_VER=7.4.1.5
NCCL_VER=2.3.7-1
BAZEL_VER=0.18.0
TF_VER=1.12.0
CONDA_VER=5.2.0
PY_VER=3.6

module load cuda/9.2.148

#install pre-built anaconda
wget https://repo.anaconda.com/archive/Anaconda3-$CONDA_VER-Linux-ppc64le.sh -O anaconda3.sh  
bash ./anaconda3.sh -b -p anaconda3 
export PATH=$(pwd)/anaconda3/bin:$PATH
pip install --upgrade pip
pip install keras_applications==1.0.4 --no-deps
pip install keras_preprocessing==1.0.2 --no-deps
#install bazel 
wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VER/bazel-$BAZEL_VER-dist.zip
unzip bazel-$BAZEL_VER-dist.zip -d bazel
cd bazel 
./compile.sh 
export PATH=$(pwd)/output:$PATH
cd ../

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
#wget  https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/v7.4.1.5/prod/9.2_20181108/cudnn-9.2-linux-ppc64le-v7.4.1.5.tgz?nebFnNYZ8NNec7aD4J4jR6_UUgIB0g4AJJb-oLhhs6qSCOYPPfOeIEmPh3aNSWFOkFsMHbTPY99JUPuKb-9QhtM9RDRj9XOwA_92awUVDlibDgh6qC9sPo6Anw6olQx2HiA4N2uGL50jX-f39Dy6Le5WXMk_q6JNX8P04iXuY4kXGyDd0ZtMGaN7jbAxjNyLxiInRpTHVA36RZ_Q-KXd2PfIx3M -O cudnn-$CUDA_VER-linux-ppc64le-v$CUDNN_VER.tgz
#mkdir cudnn-$CUDA_VER-linux-ppc64le-v$CUDNN_VER && tar -xf cudnn-$CUDA_VER-linux-ppc64le-v$CUDNN_VER.tgz -C cudnn-$CUDA_VER-linux-ppc64le-v$CUDNN_VER
#ln -s cudnn-$CUDA_VER-linux-ppc64le-v$CUDNN_VER/cuda/targets/ppc64le-linux cudnn
#ln -s lib cudnn/lib64
#CUDNN_DIR=$(pwd)/cudnn
#or use the one from cuda module 
CUDNN_DIR=$CUDA_DIR

#setup env var
export PYTHON_BIN_PATH=$(pwd)/anaconda3/bin/python3
export PYTHON_LIB_PATH=$(pwd)/anaconda3/lib/python$PY_VER/site-packages
export TF_NEED_MKL=0
export CC_OPT_FLAGS="-march=native"
export TF_NEED_JEMALLOC=1
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_ENABLE_XLA=0
export TF_NEED_OPENCL=0
export TF_NEED_CUDA=1
export TF_CUDA_CLANG=0
export TF_CUDA_VERSION=$CUDA_VER
export CUDA_TOOLKIT_PATH=$CUDA_DIR 
export TF_CUDNN_VERSION=$CUDNN_VER
export CUDNN_INSTALL_PATH=$CUDNN_DIR
export TF_CUDA_COMPUTE_CAPABILITIES="7.0"
export TF_NEED_VERBS=0
export TF_NEED_AWS=0
export TF_NEED_NGRAPH=0
export TF_NEED_S3=0
export TF_NEED_GDR=0
export TF_NEED_OPENCL_SYCL=0
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export TF_NEED_MPI=0
export TF_NEED_KAFKA=0
export TF_NEED_ROCM=0
export TF_NEED_IGNITE=0
export TF_NEED_TENSORRT=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NCCL_VERSION=$(echo $NCCL_VER | cut -d. -f1,2)
export NCCL_INSTALL_PATH="$(pwd)/nccl2"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$CUDA_DIR/lib64/stubs

#build tensorflow
git clone --recursive https://github.com/tensorflow/tensorflow
cd tensorflow 
git checkout v$TF_VER
#bug fix commit
#git cherry-pick 5aefa441
./configure
bazel --batch build --local_resources 2048,4.0,1.0 -c opt --config=cuda tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
cp /tmp/tensorflow_pkg/*.whl ../wheels




