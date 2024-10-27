# A Convolution Neural Network (CNN) From Scratch
This project implements a simple CNN network with all basic Layer for a face classification like:
- Convolution2D (only support 3x3 kernel side).
- Maxpoolng (only support 2x2 pool size).
- Flatten
- Dense (Fully Connected).
- SoftMax.
Especially, The Convolution2D and Maxpoolng was designed to run by both way: Using the CPU and using the specialized desgined FPGA module. 

## Usage
This CNN network can be used both on Personal computer and Zynq7000 SOC device. Obviously, when running on PC, you can only use the "CPU"
functionality of Convolution2D and Maxpooling layer.
Install dependencies (Required both on PC and Zynq7000):

```bash
pip install numpy
pip install matplotlib
```
**Note: on Zynq7000, you may not be able to install numpy or matplotlib by pip package, try to find a prebuilt rootfs with numpy or matplotlib 
installed or add nunpy package to rootfs by petalinux.

Then, run it with no arguments:

```bash
$ python cnn.py
$ python cnn_keras.py
```



## More

You may also be interested in [a Neural Network implemented from scratch in Python](https://github.com/vzhou842/neural-network-from-scratch), which was written for my [introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/).
