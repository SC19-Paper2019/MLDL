module load cuda
export LD_LIBRARY_PATH=$CONDA_DIR/nccl/lib:$CONDA_DIR/cudnn/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_DIR/anaconda3/bin:$PATH
