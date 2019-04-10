<b>NOTE:</b> The master is always at the latest supported version for python 3. Legacy versions have their own respective branches.<br>
<b>Latest:</b> Tensorflow 1.8, CUDA 8.0, CUDNN 7.0.5, NCCL 2.3, Python 3.5 <br>
<b>Legacy:</b> NA <br>
<b>System configuration:</b> IBM POWER8 CPU, NVIDIA Tesla P100 GPU<br> For more on summitdev go to: https://www.olcf.ornl.gov/for-users/system-user-guides/summitdev-quickstart-guide/ <br>

This repo contains the following directories:
1. benchmarks: Benchmarks for various deep learning/ machine learning models.
2. containers: Singularity containers for supported latest version.
3. documentation: Documentation for building from source, benchmarking and container building steps. 
4. install-scrips: Scripts for native installation from wheels and containers.
5. utils: Various utility scripts.
6. wheels: Python wheels for current and legacy version(s).

### Installation instructions
Installation can be done natively using python wheel as well as using singularity containers. To install the latest version follow these steps.<br>
Choose <b>only one</b> of the following options that applies:<br>

<b>Option 1.</b> A python package manager like anaconda or miniconda already installed and its `PATH` set in the system environment.<br>
Not sure? Run `conda --version` in summitdev terminal and check if the output matches with `conda <version>`. If yes then go to Option 2 else continue and  run the following commands in the terminal:
```
cd ~
wget https://code.ornl.gov/summitdev/mldl-stack/tensorflow/raw/master/install-scripts/install-tf-native.sh
chmod +x install-tf-native.sh
source install-tf-native.sh
```

<b>Option 2.</b> No python package manager is installed. Run the following commands in the terminal:
```
cd ~
wget https://code.ornl.gov/summitdev/mldl-stack/tensorflow/raw/master/install-scripts/install-tf-conda-native.sh
chmod +x install-tf-native.sh
source install-tf-conda-native.sh
```

<b>Important:</b> Don't forget to run `module load cuda/8.0.61-1` before running jsrun to run a distributed tf job. jsrun by default uses cuda 9.0 which is not compatible with this build of tf.
