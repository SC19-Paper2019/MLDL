#! /usr/bin/bash

# load modules
module load gcc/4.8.5
module load cuda/8.0.61-1

# install anaconda3
cd ~
mkdir summitdev
cd summitdev
wget https://repo.anaconda.com/archive/Anaconda3-$CONDA_VER-Linux-ppc64le.sh -O anaconda3.sh
chmod 744 anaconda3.sh
./anaconda3.sh -b -f -p $HOME/anaconda3
export PATH="$HOME/anaconda3/bin:$PATH"
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc

# copy cuda
cp -r /sw/summitdev/cuda/8.0.61-1 ~/
cd ~/8.0.61-1/nvvm/libdevice
ln -s libdevice.compute_50.10.bc libdevice.10.bc

# download cuDNN (make sure the link is login free)
cd ~
wget https://code.ornl.gov/summitdev/mldl-stack/pytorch/raw/master/dependencies/cuda.tar.gz
tar -xvzf ~/cuda.tar.gz
rm ~/cuda.tar.gz

# copy cudnn lib into cuda dir
echo copying cudnn.h ...
cp ~/cuda/targets/ppc64le-linux/include/cudnn.h ~/8.0.61-1/include
echo copying libcudnn* ...
cp ~/cuda/targets/ppc64le-linux/lib/libcudnn* ~/8.0.61-1/lib64
echo changing permissions to a+r for libcudnn*
chmod a+r ~/cuda/targets/ppc64le-linux/include/cudnn.h ~/8.0.61-1/lib64/libcudnn*

# set env variables in .bashrc
echo 'export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_TYPE=en_US.UTF-8

export HOROVOD_NCCL_HOME="/autofs/nccs-svm1_home1/$(whoami)/nccl-2.3/"

export PATH="/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1/bin:$PATH"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1/lib64"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/autofs/nccs-svm1_home1/$(whoami)/cuda/targets/ppc64le-linux/lib"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1/extras/CUPTI/lib64"

export CUDADIR="/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1"

export OPENBLASDIR="/autofs/nccs-svm1_home/$(whoami)/anaconda3"

export CMAKE_PREFIX_PATH="/ccs/home/$(whoami)/anaconda3/bin/../"

export CUDA_BIN_PATH="/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1/bin"

export CUDA_HOME="/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1"

export PYTHONPATH="${PYTHONPATH}:/autofs/nccs-svm1_home1/$(whoami)/tensorflow/benchmarks/scripts/tf_cnn_benchmarks/models"

export NCCL_ROOT_DIR="/autofs/nccs-svm1_home1/$(whoami)/nccl-2.3"

export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"

export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"

export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"

export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"

export CPATH="$NCCL_INCLUDE_DIR:$CPATH"'>> ~/source_for_tf_on_summitdev

# set env variables
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_TYPE=en_US.UTF-8

export HOROVOD_NCCL_HOME="/autofs/nccs-svm1_home1/$(whoami)/nccl-2.3/"

export PATH="/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1/bin:$PATH"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1/lib64"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/autofs/nccs-svm1_home1/$(whoami)/cuda/targets/ppc64le-linux/lib"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1/extras/CUPTI/lib64"

export CUDADIR="/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1"

export OPENBLASDIR="/autofs/nccs-svm1_home/$(whoami)/anaconda3"

export CMAKE_PREFIX_PATH="/ccs/home/$(whoami)/anaconda3/bin/../"

export CUDA_BIN_PATH="/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1/bin"

export CUDA_HOME="/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1"

export PYTHONPATH="${PYTHONPATH}:/autofs/nccs-svm1_home1/$(whoami)/tensorflow/benchmarks/scripts/tf_cnn_benchmarks/models"

export NCCL_ROOT_DIR="/autofs/nccs-svm1_home1/$(whoami)/nccl-2.3"

export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"

export LD_LIBRARY_PATH="NCCL_LIB_DIR:$LD_LIBRARY_PATH"

export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"

export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"

export CPATH="$NCCL_INCLUDE_DIR:$CPATH"

# build & install nccl
cd ~
git clone https://github.com/NVIDIA/nccl
cd nccl
git checkout v$NCCL_VER
make -j160 src.build CUDA_HOME=/autofs/nccs-svm1_home1/$(whoami)/8.0.61-1

make pkg.txz.build
cp build/pkg/txz/* ~/
cd ~
tar Jxvf nccl_*.txz

mv nccl_* nccl-2.3

cd nccl-2.3
ln -s LICENSE.txt NCCL-SLA.txt

rm -rf ~/nccl_*.txz ~/nccl 

# create virtual env
# conda create -n myenv

# env() { source $1/bin/activate myenv; }

# env anaconda3

# install tensorflow
cd ~
wget https://code.ornl.gov/summitdev/mldl-stack/tensorflow/raw/master/wheels/tensorflow-1.8.0-*.whl
pip install --disable-pip-version-check --no-cache-dir --upgrade --force-reinstall tensorflow-1.8.0-*.whl 
rm tensorflow-1.8.0-*.whl

# install horovod
cd ~
wget https://code.ornl.gov/summitdev/mldl-stack/tensorflow/raw/master/dependencies/compiler.tar.gz
tar -xzf compiler.tar.gz
mv /ccs/home/$(whoami)/compiler /ccs/home/$(whoami)/anaconda3/lib/python3.6/site-packages/tensorflow/include/tensorflow/
rm compiler.tar.gz

git clone https://github.com/uber/horovod
cd horovod
python setup.py sdist
 
HOROVOD_CUDA_HOME=~/8.0.61-1 HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=~/nccl-2.3 HOROVOD_NCCL_INCLUDE=~/nccl-2.3/include HOROVOD_NCCL_LIB=~/nccl-2.3/lib pip install --no-cache-dir dist/horovod-*.tar.gz
cd ~
rm -rf horovod

echo -e '\n'
echo -e '\tTensorflow successfully installed!!'
