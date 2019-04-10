#### These instructions install tensorflow 1.8 and NCCL 2.3 built from source using CUDA 8.0.61-1 and cuDNN 7.0.5 on summitdev. <br>
The intructions available at https://developer.ibm.com/tutorials/install-tensorflow-on-power/ have been modified specifically for summitdev. 

## Steps:
#### 1. Load required modules and execute some housekeeping commands
```
module load gcc/4.8.5
module load cuda/8.0.61-1

#set encoding to utf-8
echo 'export LANGUAGE=en_US.UTF-8' >> ~/.bashrc
echo 'export LC_ALL=en_US.UTF-8' >> ~/.bashrc
echo 'export LANG=en_US.UTF-8' >> ~/.bashrc
echo 'export LC_TYPE=en_US.UTF-8' >> ~/.bashrc

echo 'export CMAKE_PREFIX_PATH="/ccs/home/shubhankar/miniconda3/bin/../"' >> ~/.bashrc
```
#### 2. Verify NVIDIA CUDA toolkit and driver
To validate the currently installed driver and toolkit execute the following instructions:

Verify and note the CUDA version
```
$ cat /sw/summitdev/cuda/8.0.61-1/version.txt
  CUDA Version 8.0.61
```
Verify and note the driver version installed.
```
$ cat /proc/driver/nvidia/version 
NVRM version: NVIDIA UNIX ppc64le Kernel Module  390.46  Fri Mar 16 21:03:39 PDT 2018
GCC version:  gcc version 4.8.5 20150623 (Red Hat 4.8.5-16) (GCC)
```
Finally, verify that NVIDIA toolkit version is matching the CUDA version.
```
-bash-4.2$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:28:28_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```
#### 3. Copy CUDA 8.0 to home directory and set CUDA path in .bashrc
```
cp -r /sw/summitdev/cuda/8.0.61-1 ~/
```
Symlink for a bug fix
```
cd ~/8.0.61-1/nvvm/libdevice
ln -s libdevice.compute_50.10.bc libdevice.10.bc
```
Set CUDA path in .bashrc
```
export PATH="/autofs/nccs-svm1_home1/shubhankar/8.0.61-1/bin:$PATH"
export CUDADIR="/autofs/nccs-svm1_home1/shubhankar/8.0.61-1"
export CUDA_HOME="/autofs/nccs-svm1_home1/shubhankar/8.0.61-1"
export CUDA_BIN_PATH="/autofs/nccs-svm1_home1/shubhankar/8.0.61-1/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/autofs/nccs-svm1_home1/shubhankar/8.0.61-1/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/autofs/nccs-svm1_home1/shubhankar/8.0.61-1/extras/CUPTI/lib64"
export NCCL_ROOT_DIR="/autofs/nccs-svm1_home1/$(whoami)/nccl-2.3"
export NCCL_LIB_DIR="$NCCL_ROOT_DIR/lib"
export LD_LIBRARY_PATH=~"NCCL_LIB_DIR:$LD_LIBRARY_PATH"
export NCCL_INCLUDE_DIR="$NCCL_ROOT_DIR/include"
export CUDA_NVCC_EXECUTABLE="$CUDA_BIN_PATH/nvcc"
export CPATH="$NCCL_INCLUDE_DIR:$CPATH"
```
#### 4. Install CUDNN library
Go to NVIDIA cuDNN repository and download [cuDNN v7.0.5 Library for Linux (Power8)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/8.0_20171129/cudnn-8.0-linux-ppc64le-v7) (login is required) to your local machine.
Rsync the tar file from your local machine to summitdev.
```
rsync -avP ~/Downloads/cudnn-8.0-linux-ppc64le-v7.tgz shubhankar@home.ccs.ornl.gov:~/
```
Untar the .tgz file
```
tar -xvzf cudnn-8.0-linux-ppc64le-v.tgz
```
Create a symlink for a bug fix
```
cd cuda/targets/ppc64le-linux/
ln -s lib64 lib
```

Now you will have a cuDNN directory named `cuda`.
#### 5. Install Miniconda 
```
cd ~
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-ppc64le.sh
chmod 744 Miniconda3-latest-Linux-ppc64le.sh
./Miniconda3-latest-Linux-ppc64le.sh
```
Follow the instructions displayed to accept the license and set the installation path. Add the conda install location to PATH in your ~/.bashrc file.
#### 6. Create a virtual environment (optional)
Though this is an optional step, using a virtual environment will help keep your python projects isolated.
The following commands will create an environment named tf_p3.5 with Python 3.5.
```
~/miniconda3/bin/conda create -n tf_p3.5 python=3.5
source ~/miniconda3/bin/activate tf_3.5
```
#### 7. Install openBLAS and Numpy
```
conda install openblas numpy
```
#### 8. Install NCCL 2.3
```
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git checkout f93fe9b
make -j src.build CUDA_HOME=/ccs/home/shubhankar/8.0.61-1

#Install os-agnostic tarball
make pkg.txz.build
tar Jxvf build/pkg/txz/nccl_2.3.5-5+cuda8.0_ppc64le.txz -C ~/
```
Rename to nccl-2.3
```
cd ~
mv nccl_2.3.5-5+cuda8.0_ppc64le nccl-2.3
```
Bug fix
```
cd nccl-2.3 
ln -s LICENSE.txt NCCL-SLA.txt
```

#### 9. Install Bazel 0.11.1
```
cd ~
wget https://github.com/bazelbuild/bazel/releases/download/0.11.1/bazel-0.11.1-dist.zip
mkdir bazel
cd bazel
unzip ../bazel-0.11.1-dist.zip
./compile.sh
```
Copy the resulted binary under ~/bin
```
mkdir -p ~/bin
cp output/bazel ~/bin
```
Set bazel bin path in ~/.bashrc
```
echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc
```
#### 10. Install Tensorflow 1.8
```
cd ~
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout 397f04a
```
Configure TensorFlow install
```
./configure
```
Following inputs were used to the configuration questions.
```
$ ./configure
Found possible Python library paths:
  /sw/summitdev/xalt/0.7.5/site
  /ccs/home/shubhankar/miniconda3/envs/tf_p3.5/lib/python3.5/site-packages
  /autofs/nccs-svm1_home1/shubhankar/tensorflow/benchmarks/scripts/tf_cnn_benchmarks/models
  /sw/summitdev/xalt/0.7.5/libexec
Please input the desired Python library path to use.  Default is [/sw/summitdev/xalt/0.7.5/site]
/ccs/home/shubhankar/miniconda3/envs/tf_p3.5/lib/python3.5/site-packages
Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: n
No jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 8.0


Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:/ccs/home/shubhankar/8.0.61-1


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]:


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:/ccs/home/shubhankar/cuda/targets/ppc64le-linux


Do you wish to build TensorFlow with TensorRT support? [y/N]: n
No TensorRT support will be enabled for TensorFlow.

Please specify the NCCL version you want to use. [Leave empty to default to NCCL 1.3]:2.3


Please specify the location where NCCL 2 library is installed. Refer to README.md for more details. [Default is /ccs/home/shubhankar/8.0.61-1]:/ccs/home/shubhankar/nccl-2.3
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,5.2]3.5,3.7,5.2,6.0


Do you want to use clang as CUDA compiler? [y/N]: n
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:


Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -mcpu=native]: -mcpu=power8 -mtune=power8


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
        --config=mkl            # Build with MKL support.
        --config=monolithic     # Config for mostly static monolithic build.
Configuration finished
```
Build python wheel
```
bazel build -c opt --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ../tensorflow_pkg
```
Install TensorFlow
```
cd ~/tensorflow_pkg/
pip install tensorflow-1.8.0-cp35-cp35m-linux_ppc64le.whl
```
Validate installation
```
python -c "import tensorflow as tf; print(tf.__version__)"
```
