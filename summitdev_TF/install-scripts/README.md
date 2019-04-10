### Installation instructions
Installation can be done natively using python wheel as well as using singularity containers. To install the latest version follow these steps.<br>
Choose <b>only one</b> of the following options that applies:<br>

<b>Option 1.</b> A python package manager like anaconda or miniconda already installed and its `PATH` set in the system environment.<br>
Not sure? Run `conda --version` in summitdev terminal and check if the output matches with `conda <version>`. If yes then go to Option 2 else continue and  run the following commands in the terminal:
```
cd ~
wget https://code.ornl.gov/summitdev/mldl-stack/tensorflow/raw/master/install-scripts/install-tf-native.sh
chmod +x install-tf-native.sh
install-tf-native.sh
```

<b>Option 2.</b> No python package manager is installed. Run the following commands in the terminal:
```
cd ~
wget https://code.ornl.gov/summitdev/mldl-stack/tensorflow/raw/master/install-scripts/install-tf-conda-native.sh
chmod +x install-tf-conda-native.sh
install-tf-conda-native.sh
```

<b>Important:</b> Don't forget to run `module load cuda/8.0.61-1` before running jsrun to run a distributed tf job. jsrun by default uses cuda 9.0 which is not compatible with this build of tf.
