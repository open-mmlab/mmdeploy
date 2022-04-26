## 如何在 Jetson 模组上安装 mmdeploy

本教程将介绍如何将 mmdeploy 安装在 NVIDIA Jetson 模组上。主要涵盖 3 种 Jetson 模组：
- Jetson Nano
- Jetson AGX Xavier
- Jetson TX2

对于 Jetson Nano，我们使用的是 Jetson Nano 2GB，并通过 SD 卡镜像方式安装 [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)。

### 安装 JetPack SDK

主要有两种安装 JetPack 套件的方式：
1. 使用 SD 卡镜像方式，直接将镜像刻录到 SD 卡上。
2. 使用 NVIDIA SDK Manager 进行安装。

第一种方法无需两台机器以及相应的两套显示设备和连接线。我们只需按照指示刻录镜像文件即可。对于 Jetson Nano 2GB，点击 [这里](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#intro) 开始安装。对于 Jetson Nano 4GB, 点击 [这里](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) 开始安装。

第二种方法则需要我们装配另一套显示设备并连接到 Jetson 硬件上。该方法相较于第一种方法更加安全可靠，因为第一种方法有时可能会无法刻录镜像文件并在验证时发出警告。点击 [这里]((https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)) 开始。

在使用第一种方法进行安装时，如果总是提示 `Attention something went wrong...`， 并且在文件已经重新下载文件的情况下依旧显示该信息，可试着使用 wget 来下载文件并且更改 tail 命令文件名。

### 启动系统

当系统初始化过程中出现卡顿时，一般只需重启 Jetson 设备即可。

### CUDA

如果我们使用的是第一种安装 JetPack 的方法则 CUDA 是默认安装的，但 cuDNN 并不是。我们需要执行以下两行将 CUDA 路径和 lib 路径写入到环境变量 `$PATH` 和 `$LD_LIBRARY_PATH`:
```
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```
之后我们可以执行 `nvcc -V` 来获得我们使用的 CUDA 版本。

### Anaconda

我们需要安装 [Archiconda](https://github.com/Archiconda/build-tools/releases) 而不是 Anaconda，因为后者不提供针对 Jetson 的 wheel 文件。

在我们成功安装好 Archiconda 并正确创建一个虚拟环境之后，如果虚拟环境中的 pip 无法正常工作或者提示 `Illegal instruction (core dumped)`，我们可以尝试重新手动安装 pip。如果仍无法正常运行，可重新安装 JetPack SDK。

### 迁移 TensorRT 至 conda 环境

安装完 Archiconda 后，我们可以用它来创建一个虚拟环境，例如 `mmdeploy`。接着我们需要将 Jetpack 中预先安装好的 TensorRT 迁移至虚拟环境中。

首先，我们执行 `find` 来找到 TensorRT 的位置：
```
sudo find / -name tensorrt
```
然后复制 TensorRT 到目标位置，例如:
```
cp -r /usr/lib/python3.6/dist-packages/tensorrt* /home/archiconda3/env/mmdeploy/lib/python3.6/site-packages/
```
同时，Jetpack 也使得 TensorRT 的库，例如 `libnvinfer.so` 可以在 `LD_LIBRARY_PATH` 中找到。

### 安装 torch

**专门**为 Jetson 安装 PyTorch，需要点击 [这里](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) 获取 wheel 文件。此外，在执行 `pip install` 前，我们还需要安装`libopenblas-base` 和 `libopenmpi-dev`：
```
sudo apt-get install libopenblas-base libopenmpi-dev
```
否则，当我们在 python 中 import torch 时会出现如下错误：
```
libmpi_cxx.so.20: cannot open shared object file: No such file or directory
```

### 安装 torchvision

我们无法直接通过执行 `pip install torchvision` 在 Jetson Nano 上安装 torchvision。但是我们可以克隆 GitHub 上的源码并在本地编译安装。在此之前我们需要先安装一些依赖包：
```
sudo apt-get install libjpeg-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```
接着只需克隆并编译项目即可：
```
git clone git@github.com:pytorch/vision.git
cd vision
git co tags/v0.7.0 -b vision07
pip install -e .
```

### 安装 mmcv

首先安装 openSSL:
```
sudo apt-get install libssl-dev
```
然后执行 `MMCV_WITH_OPS=1 pip install -e .` 从源代码进行安装。

### 更新 CMake

这里我们以 CMake 3.20 为例：
```
sudo apt-get install -y libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
tar -zxvf cmake-3.20.0.tar.gz
cd cmake-3.20.0
./bootstrap
make
sudo make install
```
之后我们可以通过执行以下命令确认 CMake 版本：
```
source ~/.bashrc
cmake --version
```

### 安装 mmdeploy

只需按照 [这里](../build.md) 的指示进行安装。如果在安装 h5py 时提示 `failed building wheel for numpy...ERROR: Failed to build one or more wheels`，可尝试手动安装 h5py：
```
sudo apt-get install pkg-config libhdf5-100 libhdf5-dev
pip install versioned-hdf5 --no-cache-dir
```

然后我们需要手动安装 ONNX。在此之前我们要先安装 protobuf 编译器：
```
sudo apt-get install libprotobuf-dev protobuf-compiler
```
接着执行以下命令安装 ONNX:
```
pip install onnx
```
最后再重新安装 mmdeploy。
