## Access notebook running in docker from local machine

#### Prerequisite:
1. A docker image with jupyter notebook installed in it.
2. ssh access to the host hosting the container.

#### Steps:

1. ###### Create a tunnel from local machine (port 8888) to deep (port 8888) via cades-extlogin1 (port 8888):  
 >ssh -L 8885:localhost:8888 uid@cades-extlogin1.ornl.gov -t ssh -L 8888:localhost:8888 uid@deep.ornl.gov

 uid = your 3 character ucams id  
 Other ports can be used as well
2. ###### Load a jupyter enabled docker image
  1. Modify the “.bashrc” by adding following commands  
  > export docker_image=image-dash-tf-jn-keras  
  > export external_folder=/home/2sd  
  > export internal_folder=/workspace/2sd  
  > alias ltf='sudo nvidia-docker run --shm-size=1g --ulimit memlock=-1 -p 8888:8888 --ulimit stack=67108864 -it -v > $external_folder:$internal_folder $docker_image'

  2.  Load the image by typing ltf

3. ###### Start Jupyter notebook from the docker:
> jupyter notebook --no-browser --ip 0.0.0.0 --port 8888 --allow-root

4. ###### Access from your local machine’s browser:
  > localhost:8888  

  If asked about token, copy token from the jupyter notebook messages from docker
