# Distributed deep learning tutoiral on Summit 
**This tutorial only covers tool usages related to Summit, for more information on each framework please refer to corresponding documentation.** It includes steps and scripts to run distributed deep neural network (e.g. Resnet50) on ImageNet data (ILSVRC 2012) with Keras(Tensorflow backend), Pytorch, and Tensorflow on Summit. The code is from [Horovod examples](https://github.com/uber/horovod/tree/master/examples) and [Tensorflow high performance benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks), and the frameworks are based on the [PowerAI](https://developer.ibm.com/linuxonpower/deep-learning-powerai/) container. 
## Setup 
| Framework    | Version  | 
| ------------ | -------  |
| Tensorflow   | 1.10.0   |
| Pytorch      | 0.4.1    |
| Keras        | 2.2.2    |  
| Horovod      | 0.15.0   |
The PowerAI image is located at `/gpfs/wolf/stf011/world-shared/powerai` for python2 (powerai-1.5.3.1-all-ubuntu16.04-py2.simg) and python3 (powerai-1.5.3.1-all-ubuntu16.04-py3.simg), and the ImageNet data (ILSVRC 2012) is located at `/gpfs/wolf/stf011/world-shared/imagenet` on Ascent, for both JPEG format (jpeg.tar) and TFRecord (tfrecord/). 

Horovod, Keras, and tensorbaordX needs to be installed outside of the PowerAI container as follow (e.g. python2)
```bash
1. module load cuda
2. git clone https://code.ornl.gov/jqyin/mldl-hpc
3. cd mldl-hpc/tutorial 
4. cat << EOF > install-ext-dep.sh 
export PATH=/opt/anaconda2/bin:$PATH
. /opt/DL/mldl-spectrum/bin/mldl-spectrum-activate
. /opt/DL/tensorflow/bin/tensorflow-activate
. /opt/DL/pytorch/bin/pytorch-activate
pip install --install-option="--prefix=$(pwd)/external/" keras==2.2.2 
export CPATH=$CUDA_DIR/targets/ppc64le-linux/include:$CPATH
HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/usr/lib/powerpc64le-linux-gnu/ pip install --install-option="--prefix=$(pwd)/external/" horovod==0.15.0
pip install --install-option="--prefix=$(pwd)/external/" tensorboardX
export PYTHONPATH=$(pwd)/external/lib/python2.7/site-packages:$PYTHONPATH
python -c "import keras"
EOF
5. singularity exec -B /gpfs/wolf -B /autofs/nccsopen-svm1_sw/ascent -B $HOME --nv /gpfs/wolf/stf011/world-shared/powerai/powerai-1.5.3.1-all-ubuntu16.04-py2.simg bash ./install-ext-dep.sh
```
Please refer to [PowerAI on Summit](https://code.ornl.gov/jqyin/mldl-hpc/blob/master/documentation/PowerAI-on-Summit.md) for more information.  

There are several helper scripts provided under the `tutorial/utils` folder,
```bash
bench.lsf : job script for LSF scheduler 
launch.sh : script to use numactl to bind cpu and memory
run-powerai.sh : script for runtime environment and command 
setup_powerai_env.sh : script for powerai singularity container environment   
```
## Keras
[Keras](https://keras.io) is a model-level library, providing high-level building blocks for developing deep learning models. Due to its high-level abstraction on Tensorflow, CNTK, and Theano, it's well suited for quick prototyping, testing, and even production with some optimization. Following example is for Tensorflow backend. 
```bash
1. cd keras
2. wget https://raw.githubusercontent.com/uber/horovod/af69c0ffa0ab738afa33a5cfe87b88075deacf65/examples/keras_imagenet_resnet50.py
3. add measuring code via "patch -i keras.patch keras_imagenet_resnet50.py"
4. change the input format from NHWC to NCHW, which is perferable on GPU, via "sed -i 's|channels_last|channels_first|' ~/.keras/keras.json"
5. cp ../utils/bench.lsf .
6. edit jobscript "bench.lsf", and change framework to keras via "export FWK='keras'", and change charge account "PROJECT" and node counts, walltime as appropriate. Please refer to [Summit user guide](https://www.olcf.ornl.gov/for-users/system-user-guides/summit/) for more information. 
7. submit the job via "bsub bench.lsf"
```   

After the job is complete, the speed metric can be obtain via `grep "Images/sec" tst.*`, and the trainning log and the model graph can be viewed via Tensorboard as follow. Since the singularity container shares the host port, the web server can be acessed as if the Tensorbard is running natively on host. 
```bash
1. singularity shell -B /gpfs/wolf -B /autofs/nccsopen-svm1_sw/ascent --nv /gpfs/wolf/stf011/world-shared/powerai/powerai-1.5.3.1-all-ubuntu16.04-py2.simg
2. export PATH=/opt/anaconda2/bin:$PATH
3. . /opt/DL/tensorboard/bin/tensorboard-activate
4. tensorboard --logdir=logs/
5. make sure ssh tunning is working appropriately and open web browser on local machine at localhost:6006
```

To obtain the profile/trace information, other than tools provided via Tensorflow, e.g. [tfprof](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tfprof), [timeline](https://www.tensorflow.org/performance/performance_guide), for which please refer to corresponding documentation, following is the example usage of nvprof with PowerAI container. 
```bash
1. since cuda installation in PowerAI container does not come with nvprof, the job script needs to be edited to add "module load cuda; export NVPROF=$CUDA_DIR/bin/nvprof"
2. following the rest of the steps as above in a regular run. 
```
The tracing information is generated as keras-<rank>.nvprof, and can be viewed via `nvvp` tool on host. The `utils/run-powerai.sh` script can be easily modified to show other profiling information as the `nvprof` tool provides. 
 
## Pytorch 
[Pytorch](https://pytorch.org/) provides NumPy like Tensor computation and a tape-based autograd system for deep neural networks. To use Pytorch within PowerAI container,
```bash
1. cd pytorch
2. wget https://raw.githubusercontent.com/uber/horovod/af69c0ffa0ab738afa33a5cfe87b88075deacf65/examples/pytorch_imagenet_resnet50.py
3. add measuring code via "patch -i pytorch.patch pytorch_imagenet_resnet50.py"
4. cp ../utils/bench.lsf .
5. edit bench.lsf jobscript similarly as in above Keras example.
6. bsub bench.lsf. 
```
The job will generate checkpints and log files that can be view by Tensorboard too. To get the speed metric, try `grep "Train Images/sec" tst.*`.
## Tensorflow 
[Tensorflow](https://www.tensorflow.org/) is Google's open-source machine learning framework. To use Tensorflow within PowerAI container,
```bash
1. cd tf
2. singularity shell -B /gpfs/wolf -B /autofs/nccsopen-svm1_sw/ascent --nv /gpfs/wolf/stf011/world-shared/powerai/powerai-1.5.3.1-all-ubuntu16.04-py2.simg
3. /opt/DL/tensorflow-performance-models/bin/tensorflow-install-models .
4. cp ../utils/bench.lsf .
5. edit bench.lsf jobscript similarly as in above Keras example.
6. bsub bench.lsf. 
```
## Ray 
[Ray](https://ray.readthedocs.io) is a distributed execution engine for reinforcement learning and hyperparameter tuning. 

A full example including Summit job script and scripts to start/stop a Ray cluster on demand for population based training of ResNet can be found at `tutorial/utils/ray`.
