#!/bin/bash
nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
head=${nodes[0]}
unset nodes[0] 

debug="${RAY_DEBUG:-0}"
if [ $debug -eq 0 ]; then 
  tmpdir=/tmp
else
  tmpdir=$(pwd)/tmp
fi 
ssh $head PYTHONPATH=$PYTHONPATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH ray start --head --no-ui --redis-port=6379 --temp-dir=$tmpdir --num-cpus=42 --num-gpus=6
if [ $? -eq 0 ]; then
  echo "Ray head node started on $head"
else 
  echo "Ray head node failed to start"
  exit 1
fi 

for worker in ${nodes[@]}
do
  ssh $worker PYTHONPATH=$PYTHONPATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH ray start --redis-address="$head:6379" --temp-dir=$tmpdir --num-cpus=42 --num-gpus=6  &
  if [ $? -eq 0 ]; then
    echo "Ray worker started on $worker"
  fi 
done 
wait 

echo "Sanity check:" 
jsrun -n1 -a1 -c1 -r1 python cluster-check.py $head 

export RAY_MASTER=$head 
 
