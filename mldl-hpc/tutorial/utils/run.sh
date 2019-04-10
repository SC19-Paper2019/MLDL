#!/bin/bash
if [ ! -z "$NVPROF" ]
then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_DIR/lib64
  NVPROF+=" -o keras-${PMIX_RANK}.nvprof"
fi
case "$FWK" in 
"keras") 
  if [ "$MODE" = "powerai" ]; then  
  . /opt/DL/tensorflow/bin/tensorflow-activate
  fi
  $NVPROF python -u keras_imagenet_resnet50.py  --train-dir=$DATADIR/train  --val-dir=$DATADIR/validation  --epochs=10 --batch-size=64 --val-batch-size=64
;;
"pytorch")  
  if [ "$MODE" = "powerai" ]; then  
  . /opt/DL/pytorch/bin/pytorch-activate
  fi
  $NVPROF python -u pytorch_imagenet_resnet50.py --train-dir=$DATADIR/train  --val-dir=$DATADIR/validation  --epochs=10 --batch-size=64 --val-batch-size=64
;;
"tensorflow")
  if [ "$MODE" = "powerai" ]; then  
  . /opt/DL/tensorflow/bin/tensorflow-activate
  fi
  $NVPROF python -u tf_cnn_benchmarks/tf_cnn_benchmarks.py --variable_update=horovod --model=resnet50  --num_gpus=1 --batch_size=64 --num_batches=1000 --num_warmup_batches=10 --data_dir=$DATADIR --data_name=imagenet
;;
*)
echo "unsupported framework: $FWK"
;;
esac 
