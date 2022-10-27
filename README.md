# Introduction
This is an implementation of SRGAN with pytorch lightning. I code this for learning pytorch lightning and SRGAN.

# Training
In my training, I use DIV2K and VOC2012 dataset. You can create your own dataset directory and edit the path in your config yaml file.

# How to use
## Environment setup
You can install the conda environment with `environment.yml` provided. Command:
> conda env create -f environment.yml
## Train your own model
In `src/config`, you can edit or create your own config yaml file. This project uses [hydra](https://hydra.cc/) to enable training configuration with yaml. The `debug.yaml` is for testing your environment. \
After you have your own config yaml file, you can run my code with command: 
>python train.py --config-name {your config file name}