## How to install mmdeploy on Jetsons

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
```shell
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```
Then we can use `nvcc -V` the get the version of cuda we use.

### Conda

We have to install [Archiconda](https://github.com/Archiconda/build-tools/releases) instead as the Anaconda does not provide the wheel built for jetson.
```shell
wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh
bash Archiconda3-0.2.3-Linux-aarch64.sh
source ~/.bashrc
conda --version
```

After we installed the Archiconda successfully, we can creat the virtual env `mmdeploy` using the command below. 
```shell
conda create -n mmdeploy python=3.6  # must be python 3.6
conda activate mmdeploy
```

### Move tensorrt to conda env
Then we have to move the pre-installed tensorrt package in Jetpack to the virtual env.

First we use `find` to get where the tensorrt is
```shell
sudo find / -name tensorrt
```
Then copy the tensorrt to our destination like:
```shell
cp -r /usr/lib/python3.6/dist-packages/tensorrt* /your/path/to/archiconda3/envs/mmdeploy/lib/python3.6/site-packages/
```
Meanwhile, tensorrt libs like `libnvinfer.so` can be found in `LD_LIBRARY_PATH`, which is done by Jetpack as well.

### Install PyTorch

Before we use `pip install`, we have to install `libopenblas-base`, `libopenmpi-dev` first:
```shell
sudo apt-get install -y libopenblas-base libopenmpi-dev
```
Or, it will throw the following error when we import torch in python:
```
libmpi_cxx.so.20: cannot open shared object file: No such file or directory
```
After that, download the PyTorch wheel for Jetson **specifically**. Click [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) to get the wheel. MMDeploy is using `pytorch==1.8.0`.

After the download finished, using cmd to install it.
```shell
pip3 install /your/path/dwonload/xxx.whl
```

### Install torchvision
We can't directly use `pip install torchvision` to install torchvision for Jetson Nano. But we can clone the repository from Github and build it locally. First we have to install some dependencies:
```shell
sudo apt-get install -y libjpeg-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```
Then just clone and compile the project:
```shell
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.9.0
pip install -e .
```

**Note:** If the `pip install` throws error: `Illegal instruction (core dumped)`, then add `export OPENBLAS_CORETYPE=ARMV8` in `~/.bashrc` and then `source ~/.bashrc && conda activate mmdeploy` will solve the problem.

### Install mmcv

Install openssl first:
```shell
sudo apt-get install -y libssl-dev
```
Then install it from source, MMDeploy using mmcv version is `1.4.0`
```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.4.0
MMCV_WITH_OPS=1 pip install -e .
```

### Update cmake

We choose cmake version 20 as an example.
```shell
sudo apt remove cmake
sudo apt purge --auto-remove cmake
sudo apt-get install -y libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
tar -zxvf cmake-3.20.0.tar.gz
cd cmake-3.20.0
./bootstrap
make -j$(nproc)
sudo make install
```
Then we can check the cmake version through:
```shell
source ~/.bashrc
cmake --version
```

### Install spdlog
```shell
sudo apt-get install libspdlog-dev
```

### Install onnxruntime

1. Install python package
```shell
pip install onnxruntime==1.8.1
```

2. Download the linux prebuilt binary package from [here](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1).  Extract it and export environment variables as below:
```shell
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

### Install pplcv
A high-performance image processing library of openPPL. Now, MMDeploy supports `v0.6.2` and has to use git clone to download it.
```shell
git clone https://github.com/openppl-public/ppl.cv.git
cd ppl.cv
export PPLCV_DIR=$(pwd)
git checkout tags/v0.6.2 -b v0.6.2
./build.sh cuda
```

### Install h5py
```shell
sudo apt-get install pkg-config libhdf5-100 libhdf5-dev
pip install versioned-hdf5 --no-cache-dir
```

### Install MMDeploy
Using git to clone MMDeploy source code.
```shell
git clone -b master https://github.com/open-mmlab/mmdeploy.git MMDeploy
cd MMDeploy
git submodule update --init --recursive
````

Build MMDeploy from source:
```shell
mkdir -p build && cd build
cmake .. \
    -DCMAKE_CXX_COMPILER=g++-7 \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    -DMMDEPLOY_TARGET_BACKENDS="ort,trt" \
    -DMMDEPLOY_CODEBASES=all \
    -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl \
    -DTENSORRT_DIR=/usr/src/tensorrt \
    -DCUDNN_DIR=/etc/alternatives

make -j$(nproc) && make install
```

### Install MMDeploy Python API 

```shell
cd /path/to/mmdeploy
pip install -e .
```

### FAQs

- For Jetson TX2 and Jetson Nano, `#assertion/root/workspace/mmdeploy/csrc/backend_ops/tensorrt/batched_nms/trt_batched_nms.cpp,98` or `pre_top_k need to be reduced for devices with arch 7.2`

    Set MAX N mode and `sudo nvpmodel -m 0 && sudo jetson_clocks`.
    Reducing the number of [pre_top_k](https://github.com/open-mmlab/mmdeploy/blob/34879e638cc2db511e798a376b9a4b9932660fe1/configs/mmdet/_base_/base_static.py#L13) to reduce the number of proposals may resolve the problem.
