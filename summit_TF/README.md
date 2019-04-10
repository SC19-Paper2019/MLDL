<b>System configuration:</b> IBM POWER9 CPU, NVIDIA Volta V100 GPU<br> For more on summit go to: https://www.olcf.ornl.gov/for-users/system-user-guides/summit/ <br>

This repo contains the following directories:
1. benchmarks: Benchmarks for various deep learning/ machine learning models.
2. containers: Singularity containers for supported latest version.
3. documentation: Documentation for building from source, benchmarking and container building steps. 
4. install-scrips: Scripts for native installation from wheels and containers.
5. utils: Various utility scripts.
6. wheels: Python wheels for current and legacy version(s).

### Available versions:
|              | **Tensorflow** | 
|--------------| :-------------: | 
| **Python 3** | 1.12.0, 1.11.0, 1.8.0  | 
| **Python 2** | 1.12.0, 1.11.0, 1.8.0  |

### Installation instructions:
For native installation run the following commands. Requirements will be asked during installation.<br>

```
wget https://code.ornl.gov/summit/mldl-stack/tensorflow/raw/master/install-scripts/install-tf-native.sh
chmod +x install-tf-native.sh
source install-tf-native.sh
```

**NOTE:** This creates a tensorflow directory with specific version inside a summit directory. `source` the source file created inside the tensorflow directory before running python and importing tensorflow.
