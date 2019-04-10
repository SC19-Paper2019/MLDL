Benchmarking for renset50 was done using the native installation of tensorflow 1.8, CUDA 8.0, CUDNN 7.0.5 and NCCL 2.3 with a batch size of 64 using horovod which was also built from source. The following jsrun configuration was used:<br>
`jsrun -n<number of nodes> -a4 -c20 -g4 python ~/tensorflow/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --variable_update horovod`<br>
The bechmarks are available [here](https://code.ornl.gov/summitdev/mldl-stack/tensorflow/blob/master/benchmarks/resnet50-benchmarks.png).

