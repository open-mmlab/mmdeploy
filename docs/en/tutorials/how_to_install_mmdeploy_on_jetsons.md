## How to install mmdeploy on Jetsons

This tutorial introduces how to install mmdeploy on Nvidia Jetson systems. It mainly introduces the installation of mmdeploy on three Jetson series boards:
- Jetson Nano
- Jetson AGX Xavier
- Jetson TX2

For Jetson Nano, we use Jetson Nano 2GB and install [JetPack SDK](https://developer.nvidia.com/embedded/jetpack) through SD card image method.

**Note**: The JetPack we use is `4.6.1`, and the default python version of it is `3.6`.

### Install JetPack SDK

There are mainly two ways to install the JetPack:
1. Write the image to the SD card directly.
2. Use the SDK Manager to do this.

The first method does not need two separated machines and their display equipment or cables. We just follow the instruction to write the image. This is pretty convenient. Click [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#intro) for Jetson Nano 2GB to start. And click [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) for Jetson Nano 4GB to start the journey.

The second method, however, requires we set up another display tool and cable to the jetson hardware. This method is safer than the previous one as the first method may sometimes cannot write the image in and throws a warning during validation. Click [here](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html) to start.

For the first method, if it always throws `Attention something went wrong...` even the file already get re-downloaded, just try `wget` to download the file and change the tail name instead.

### Launch the system

Sometimes we just need to reboot the jetson device when it gets stuck in initializing the system.

### CUDA

The CUDA is installed by default and we have to write the CUDA path and lib to `$PATH` and `$LD_LIBRARY_PATH`:
```shell
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```
Then we can use `nvcc -V` the get the version of cuda we use.

If you want to save CUDA env variables to bashrc, you could run:
```bash
echo '# set env for CUDA' >> ~/.bashrc
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
source ~/.bashrc
nvcc -V
```

### pip env
When we using `pip install` to install some package, it will throws error: `Illegal instruction (core dumped)`
Follow the below step will solve the problem:
```shell
echo '# set env for TensorRT' >> ~/.bashrc
echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
source ~/.bashrc
source activate mmdeploy
```

### Conda

We have to install [Archiconda](https://github.com/Archiconda/build-tools/releases) instead as the Anaconda does not provide the wheel built for jetson. The commands below are the example for installation. You can choose another version of Archiconda by accessing [Archiconda releases page](https://github.com/Archiconda/build-tools/releases).
```shell
wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh
bash Archiconda3-0.2.3-Linux-aarch64.sh -b -p ~/conda
rm Archiconda3-0.2.3-Linux-aarch64.sh
echo '# set env for conda' >> ~/.bashrc
echo 'export PATH=$PATH:~/conda/bin' >> ~/.bashrc
source ~/.bashrc
conda --version
```

After we install the Archiconda successfully, we need to check the tensorrt python package pre-installed by Jetpack by command below.
```shell
sudo find / -name tensorrt
```
Then you can see something like those in the terminal. Take Jetson Nano as example:
```shell
...
/usr/lib/python3.6/dist-packages/tensorrt
...
```
The `python3.6` is the version we need to use in the conda env later.

We can create the virtual env `mmdeploy` using the command below. Ensure the python version in the command is the same as the above.
```shell
conda create -y -n mmdeploy python=3.6
```

### Make TensorRT available in your Conda env
Then we have to move the pre-installed tensorrt package in Jetpack to the virtual env.

First we use `find` to get where the tensorrt is
```shell
sudo find / -name tensorrt
```
Then copy the tensorrt to our destination like:
```shell
cp -r /usr/lib/python3.6/dist-packages/tensorrt* ~/conda/envs/mmdeploy/lib/python3.6/site-packages/
```
Meanwhile, tensorrt libs like `libnvinfer.so` can be found in `LD_LIBRARY_PATH`, which is done by Jetpack as well.

Final command of this step: export `TENSORRT_DIR` to the system env for MMDeploy installation:
```shell
echo '# set env for TensorRT' >> ~/.bashrc
echo 'export TENSORRT_DIR=/usr/include/aarch64-linux-gnu' >> ~/.bashrc
source ~/.bashrc
```
Then we can activate mmdeploy env for test it.
```shell
source activate mmdeploy
python -c "import tensorrt; print(tensorrt.__version__)" # Will print the vresion of TensorRT
```

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
pip3 install /your/path/dwonload/torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```

### Install torchvision
We can't directly use `pip install torchvision` to install torchvision for Jetson Nano. But we can clone the repository from Github and build it locally. First we have to install some dependencies:
```shell
sudo apt-get install -y libjpeg-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```
Then just clone and compile the project, MMDeploy is using `torchvision=0.9.0`:
```shell
git clone https://github.com/pytorch/vision.git
cd vision
git checkout tags/v0.9.0 -b v0.9.0
pip install -e .
```

**Note:** If the `pip install` throws error: `Illegal instruction (core dumped)`, then add `export OPENBLAS_CORETYPE=ARMV8` in `~/.bashrc` and then `source ~/.bashrc && conda activate mmdeploy` will solve the problem.

### Install mmcv

Install openssl first:
```shell
sudo apt-get install -y libssl-dev
```
Since MMDeploy is using mmcv 1.4.0, you can install it from source as below:
```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.4.0
MMCV_WITH_OPS=1 pip install -e .
```

### Update cmake

We choose cmake version `v3.23.1` as an example. We use the pre-built cmake binary to update it.

| Install type |
| --- |
| [cmake-3.23.1-linux-aarch64.sh](https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1-linux-aarch64.sh) |
|[cmake-3.23.1-linux-aarch64.tar.gz](https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1-linux-aarch64.tar.gz)|

### Install spdlog
```shell
sudo apt-get install libspdlog-dev
```

### Install onnx
```shell
pip install onnx
```

### Install pplcv
PPL.CV is a high-performance image processing library of OpenPPL. We need to use git clone to download it.
```shell
git clone https://github.com/openppl-public/ppl.cv.git
cd ppl.cv
export PPLCV_DIR=$(pwd)
./build.sh cuda
```

### Install h5py
```shell
sudo apt-get install pkg-config libhdf5-100 libhdf5-dev
pip install versioned-hdf5
```

### Install MMDeploy
Using git to clone MMDeploy source code.
```shell
git clone -b master https://github.com/open-mmlab/mmdeploy.git MMDeploy
cd MMDeploy
export MMDEPLOY_DIR=$(pwd)
git submodule update --init --recursive
```

Build MMDeploy from source:
```shell
mkdir -p build && cd build
cmake .. \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    -DMMDEPLOY_TARGET_BACKENDS="trt" \
    -DMMDEPLOY_CODEBASES=all \
    -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl

make -j$(nproc) && make install
```

### Install MMDeploy Python API

```shell
cd ${MMDEPLOY_DIR}
pip install -e .
```

### Build MMDeploy SDK Example

```shell
cd ${MMDEPLOY_DIR}/build/install/example
mkdir -p build && cd build
cmake .. -DMMDeploy_DIR=${MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy
make -j$(nproc)
```

running the object detection example:
```shell
./object_detection cuda ${work_dir} ${path/to/an/image}
```

### FAQs

- **F-1**: For **Jetson TX2** and **Jetson Nano**, may get the error: `#assertion/root/workspace/mmdeploy/csrc/backend_ops/tensorrt/batched_nms/trt_batched_nms.cpp,98` or `pre_top_k need to be reduced for devices with arch 7.2`.

  **Q**: There 2 steps you need to do:
  1. Set `MAX N` mode and process `sudo nvpmodel -m 0 && sudo jetson_clocks`.
  2. Reducing the number of `pre_top_k` in deploy config file like [mmedt pre_top_k](https://github.com/open-mmlab/mmdeploy/blob/34879e638cc2db511e798a376b9a4b9932660fe1/configs/mmdet/_base_/base_static.py#L13) to reduce the number of proposals may resolve the problem. I reduce it to `1000` and it work.

- **F-2**: `pip install` throws error: `Illegal instruction (core dumped)`
  **Q**: Follow the below step will solve the problem:
  ```shell
  echo '# set env for TensorRT' >> ~/.bashrc
  echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
  source ~/.bashrc
  source activate mmdeploy
  ```
