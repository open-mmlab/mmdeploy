# Build for Jetson

In this chapter, we introduce how to install MMDeploy on NVIDIA Jetson platforms, which we have verified on the following modules:
- Jetson Nano
- Jetson TX2
- Jetson AGX Xavier

## Prerequisites

To equip a Jetson device, the JetPack SDK is a must.
Besides, the Model Converter of MMDeploy requires an environment with PyTorch for converting PyTorch models to ONNX models.
Regarding the toolchain, CMake and GCC has to be upgraded to no less than 3.14 and 7.0 respectively.

### JetPack SDK

JetPack SDK provides a full development environment for hardware-accelerated AI-at-the-edge development.
All Jetson modules and developer kits are supported by JetPack SDK.

There are two major installation methods including,
1. SD Card Image Method
2. NVIDIA SDK Manager Method

You can find a very detailed installation guide from NVIDIA [official website](https://developer.nvidia.com/jetpack-sdk-50dp).

Here we choose [JetPack 4.6.1](https://developer.nvidia.com/jetpack-sdk-461) as our best practice on setup Jetson platforms. MMDeploy has been tested on JetPack 4.6 rev3 and above and TensorRT 8.0.1.6 and above. Earlier JetPack versions has incompatibilities with TensorRT 7.x

### Conda

Install [Archiconda](https://github.com/Archiconda/build-tools/releases) instead of Anaconda because the latter does not provide the wheel built for Jetson.

```shell
wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh
bash Archiconda3-0.2.3-Linux-aarch64.sh -b

echo -e '\n# set environment variable for conda' >> ~/.bashrc
echo ". ~/archiconda3/etc/profile.d/conda.sh" >> ~/.bashrc
echo 'export PATH=$PATH:~/archiconda3/bin' >> ~/.bashrc

echo -e '\n# set environment variable for pip' >> ~/.bashrc
echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc

source ~/.bashrc
conda --version
```

After the installation, create a conda environment and activate it.

```shell
# get the version of python3 installed by default
export PYTHON_VERSION=`python3 --version | cut -d' ' -f 2 | cut -d'.' -f1,2`
conda create -y -n mmdeploy python=${PYTHON_VERSION}
conda activate mmdeploy
```

```{note}
JetPack SDK 4+ provides python 3.6. We strongly recommend using the default python. Trying to upgrade it probably will ruin the JetPack environment.

If a higher-version python is necessary, you can install JetPack 5+, in which the python version is 3.8.
```
### PyTorch

Download the PyTorch wheel for Jetson from [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and save it to the local directory `/opt`.
And build torchvision from source as there is no prebuilt torchvision for Jetson platforms.

Take `torch 1.10.0` and  `torchvision 0.11.1` for example. You can install them as below:

```shell
# pytorch
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
# torchvision
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev -y
sudo rm -r torchvision
git clone https://github.com/pytorch/vision torchvision
cd torchvision
git checkout tags/v0.11.1 -b v0.11.1
export BUILD_VERSION=0.11.1
pip install -e .
```

If you install other versions of PyTorch and torchvision, make sure the versions are compatible. Refer to the compatibility chart listed [here](https://pypi.org/project/torchvision/).

### CMake

We use the latest cmake v3.23.1 released in April 2022.

```shell
# purge existing
sudo apt-get purge cmake
sudo snap remove cmake

# install prebuilt binary
export CMAKE_VER=3.23.1
export ARCH=aarch64
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-${ARCH}.sh
chmod +x cmake-${CMAKE_VER}-linux-${ARCH}.sh
sudo ./cmake-${CMAKE_VER}-linux-${ARCH}.sh --prefix=/usr --skip-license
cmake --version
```

## Install Dependencies

The Model Converter of MMDeploy on Jetson platforms depends on [MMCV](https://github.com/open-mmlab/mmcv) and the inference engine [TensorRT](https://developer.nvidia.com/tensorrt).
While MMDeploy C/C++ Inference SDK relies on [spdlog](https://github.com/gabime/spdlog), OpenCV and [ppl.cv](https://github.com/openppl-public/ppl.cv) and so on， as well as TensorRT.
Thus, in the following sections, we will describe how to prepare TensorRT.
And then, we will present the way to install dependencies of Model Converter and C/C++ Inference SDK respectively.

### Prepare TensorRT

TensorRT is already packed into JetPack SDK. However, in order to import it successfully in the conda environment,
we need to copy the tensorrt package to the conda environment created before.

```shell
cp -r /usr/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* ~/archiconda3/envs/mmdeploy/lib/python${PYTHON_VERSION}/site-packages/
conda deactivate
conda activate mmdeploy
python -c "import tensorrt; print(tensorrt.__version__)" # Will print the version of TensorRT

# set environment variable for building mmdeploy later on
export TENSORRT_DIR=/usr/include/aarch64-linux-gnu

# append cuda path and libraries to PATH and LD_LIBRARY_PATH, which is also used for building mmdeploy later on
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

You can also make the above environment variables permanent by adding them to `~/.bashrc`.

```shell
echo -e '\n# set environment variable for TensorRT' >> ~/.bashrc
echo 'export TENSORRT_DIR=/usr/include/aarch64-linux-gnu' >> ~/.bashrc

echo -e '\n# set environment variable for CUDA' >> ~/.bashrc
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc

source ~/.bashrc
conda activate mmdeploy
```

### Install Dependencies for Model Converter

- Install [MMCV](https://github.com/open-mmlab/mmcv)

  MMCV hasn't provided prebuilt package for Jetson platforms, so we have to build it from source.

  ```shell
  sudo apt-get install -y libssl-dev
  git clone https://github.com/open-mmlab/mmcv.git
  cd mmcv
  git checkout v1.4.0
  MMCV_WITH_OPS=1 pip install -e .
  ```

- Install ONNX

  ```shell
  pip install onnx
  ```

- Install h5py

  Model Converter employs HDF5 to save the calibration data for TensorRT INT8 quantization.

  ```shell
  sudo apt-get install -y pkg-config libhdf5-100 libhdf5-dev
  pip install versioned-hdf5
  ```

### Install Dependencies for SDK

You can skip this section if you don't need MMDeploy C/C++ Inference SDK.

- Install [spdlog](https://github.com/gabime/spdlog)

  "`spdlog` is a very fast, header-only/compiled, C++ logging library"

  ```shell
  sudo apt-get install -y libspdlog-dev
  ```

- Install [ppl.cv](https://github.com/openppl-public/ppl.cv)

  "`ppl.cv` is a high-performance image processing library of [OpenPPL](https://openppl.ai/home)"

  ```shell
  git clone https://github.com/openppl-public/ppl.cv.git
  cd ppl.cv
  export PPLCV_DIR=$(pwd)
  echo -e '\n# set environment variable for ppl.cv' >> ~/.bashrc
  echo "export PPLCV_DIR=$(pwd)" >> ~/.bashrc
  ./build.sh cuda
  ```

## Install MMDeploy

```shell
git clone --recursive https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
export MMDEPLOY_DIR=$(pwd)
```

### Install Model Converter

Since some operators adopted by OpenMMLab codebases are not supported by TenorRT,
we build the custom TensorRT plugins to make it up, such as `roi_align`, `scatternd`, etc.
You can find a full list of custom plugins from [here](../ops/tensorrt.md).

```shell
# build TensorRT custom operators
mkdir -p build && cd build
cmake .. -DMMDEPLOY_TARGET_BACKENDS="trt"
make -j$(nproc)

# install model converter
cd ${MMDEPLOY_DIR}
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without re-installation.
```

### Install C/C++ Inference SDK

You can skip this section if you don't need MMDeploy C/C++ Inference SDK.

1. Build SDK Libraries

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

2. Build SDK demos

    ```shell
    cd ${MMDEPLOY_DIR}/build/install/example
    mkdir -p build && cd build
    cmake .. -DMMDeploy_DIR=${MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy
    make -j$(nproc)
    ```

3. Run a demo

    Take the object detection for example:
    ```shell
    ./object_detection cuda ${directory/to/the/converted/models} ${path/to/an/image}
    ```

## Troubleshooting

### Installation

- `pip install` throws an error like `Illegal instruction (core dumped)`

  ```shell
  echo '# set env for pip' >> ~/.bashrc
  echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
  source ~/.bashrc
  ```

  If the steps above don't work, check if you are using any mirror. If so, try this:
  ```shell
  rm .condarc
  conda clean -i
  conda create -n xxx python=${PYTHON_VERSION}
  ```

### Runtime

- `#assertion/root/workspace/mmdeploy/csrc/backend_ops/tensorrt/batched_nms/trt_batched_nms.cpp,98` or `pre_top_k need to be reduced for devices with arch 7.2`

  1. Set `MAX N` mode and perform `sudo nvpmodel -m 0 && sudo jetson_clocks`.
  2. Reduce the number of `pre_top_k` in deploy config file like [mmdet pre_top_k](https://github.com/open-mmlab/mmdeploy/blob/34879e638cc2db511e798a376b9a4b9932660fe1/configs/mmdet/_base_/base_static.py#L13) does, e.g., `1000`.
  3. Convert the model again and try SDK demo again.
