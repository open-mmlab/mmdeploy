## Build for Jetson

This tutorial introduces how to install mmdeploy on Nvidia Jetson systems. It mainly introduces the installation of mmdeploy on three Jetson series boards:
- Jetson Nano
- Jetson AGX Xavier
- Jetson TX2

For Jetson Nano, we use Jetson Nano 2GB and install [JetPack SDK](https://developer.nvidia.com/embedded/jetpack) through SD card image method.

### Install JetPack SDK

There are mainly two ways to install the JetPack:
1. Write the image to the SD card directly.
2. Use the SDK Manager to do this.

The first method does not need two separated machines and their display equipment or cables. We just follow the instruction to write the image. This is pretty convenient. Click [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#intro) for Jetson Nano 2GB to start. And click [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) for Jetson Nano 4GB to start the journey.

The second method, however, requires we set up another display tool and cable to the jetson hardware. This method is safer than the previous one as the first method may sometimes cannot write the image in and throws a warning during validation. Click [here](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html) to start.

For the first method, if it always throws `Attention something went wrong...` even the file already get re-downloaded, just try `wget` to download the file and change the tail name instead.

### Launch the system

Sometimes we just need to reboot the jetson device when it gets stuck in initializing the system.

### Cuda

The Cuda is installed by default while the cudnn is not if we use the first method. We have to write the cuda path and lib to `$PATH` and `$LD_LIBRARY_PATH`:
```
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```
Then we can use `nvcc -V` the get the version of cuda we use.

### Anaconda

We have to install [Archiconda](https://github.com/Archiconda/build-tools/releases) instead as the Anaconda does not provide the wheel built for jetson.

After we installed the Archiconda successfully and created the virtual env correctly. If the pip in the env does not work properly or throw `Illegal instruction (core dumped)`, we may consider re-install the pip manually, reinstalling the whole JetPack SDK is the last method we can try.

### Move tensorrt to conda env
After we installed the Archiconda, we can use it to create a virtual env like `mmdeploy`. Then we have to move the pre-installed tensorrt package in Jetpack to the virtual env.

First we use `find` to get where the tensorrt is
```
sudo find / -name tensorrt
```
Then copy the tensorrt to our destination like:
```
cp -r /usr/lib/python3.6/dist-packages/tensorrt* /home/archiconda3/env/mmdeploy/lib/python3.6/site-packages/
```
Meanwhle, tensorrt libs like `libnvinfer.so` can be found in `LD_LIBRARY_PATH`, which is done by Jetpack as well.

### Install torch

Install the PyTorch for Jetsons **specifically**. Click [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) to get the wheel. Before we use `pip install`, we have to install `libopenblas-base`, `libopenmpi-dev` first:
```
sudo apt-get install libopenblas-base libopenmpi-dev
```
Or, it will throw the following error when we import torch in python:
```
libmpi_cxx.so.20: cannot open shared object file: No such file or directory
```

### Install torchvision
We can't directly use `pip install torchvision` to install torchvision for Jetson Nano. But we can clone the repository from Github and build it locally. First we have to install some dependencies:
```
sudo apt-get install libjpeg-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```
Then just clone and compile the project:
```
git clone git@github.com:pytorch/vision.git
cd vision
git co tags/v0.7.0 -b vision07
pip install -e .
```

### Install mmcv

Install openssl first:
```
sudo apt-get install libssl-dev
```
Then install it from source like `MMCV_WITH_OPS=1 pip install -e .`

### Update cmake

We choose cmake version 20 as an example.
```
sudo apt-get install -y libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
tar -zxvf cmake-3.20.0.tar.gz
cd cmake-3.20.0
./bootstrap
make
sudo make install
```
Then we can check the cmake version through:
```
source ~/.bashrc
cmake --version
```

### Install mmdeploy
Just follow the instruction [here](../build.md). If it throws `failed building wheel for numpy...ERROR: Failed to build one or more wheels` when installing `h5py`, try install `h5py` manually.
```
sudo apt-get install pkg-config libhdf5-100 libhdf5-dev
pip install versioned-hdf5 --no-cache-dir
```

Then install onnx manually. First, we have to install protobuf compiler:
```
sudo apt-get install libprotobuf-dev protobuf-compiler
```
Then install onnx through:
```
pip install onnx
```
Then reinstall mmdeploy.


### FAQs

- For Jetson TX2 and Jetson Nano, `#assertion/root/workspace/mmdeploy/csrc/backend_ops/tensorrt/batched_nms/trt_batched_nms.cpp,98` or `pre_top_k need to be reduced for devices with arch 7.2`

    Set MAX N mode and `sudo nvpmodel -m 0 && sudo jetson_clocks`.
    Reducing the number of [pre_top_k](https://github.com/open-mmlab/mmdeploy/blob/34879e638cc2db511e798a376b9a4b9932660fe1/configs/mmdet/_base_/base_static.py#L13) to reduce the number of proposals may resolve the problem.
