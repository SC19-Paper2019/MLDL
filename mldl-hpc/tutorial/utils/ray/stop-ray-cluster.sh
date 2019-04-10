#!/bin/bash
nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))

for worker in ${nodes[@]}
do
  ssh $worker PYTHONPATH=$PYTHONPATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH ray stop  &
  if [ $? -eq 0 ]; then
    echo "Ray stopped on $worker"
  fi 
done 
wait 


