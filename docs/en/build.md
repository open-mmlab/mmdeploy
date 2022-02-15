# Build MMDeploy

- [Build MMDeploy](#build-mmdeploy)
  - [Download MMDeploy](#download-mmdeploy)
  - [Build for Linux-x86_64](#build-for-linux-x86_64)
    - [1. Dockerfile (RECOMMENDED)](#1-dockerfile-recommended)
    - [2. Build From Source](#2-build-from-source)
      - [Install Toolchains](#install-toolchains)
      - [Install Dependencies for Model Converter](#install-dependencies-for-model-converter)
      - [Install Dependencies for SDK](#install-dependencies-for-sdk)
      - [Install Inference Engines for MMDeploy](#install-inference-engines-for-mmdeploy)
        - [ONNXRuntime](#onnxruntime)
        - [TensorRT](#tensorrt)
        - [ncnn](#ncnn)
        - [pplnn](#pplnn)
        - [OpenVINO](#openvino)
      - [Build Model Converter](#build-model-converter)
      - [Build SDK](#build-sdk)
    - [3. Use Prebuit Package](#3-use-prebuit-package)
  - [Build for Windows-x86_64](#build-for-windows-x86_64)
    - [1. Build From Source](#1-build-from-source)
    - [2. Use Prebuit Package](#2-use-prebuit-package)
## Download MMDeploy

    ```bash
    git clone -b master git@github.com:open-mmlab/mmdeploy.git MMDeploy
    cd MMDeploy
    git submodule update --init --recursive
    ```


Note: If fetching submodule fails, you could get submodule manually by following instructions:

      ```bash
      git clone git@github.com:NVIDIA/cub.git third_party/cub
      cd third_party/cub
      git checkout c3cceac115

      # go back to third_party directory and git clone pybind11
      cd ..
      git clone git@github.com:pybind/pybind11.git pybind11
      cd pybind11
      git checkout 70a58c5
      ```

## Build for Linux-x86_64

MMDeploy provides two build ways under linux-x86_64 platform, including dockerfile and building from source.

### 1. Dockerfile (RECOMMENDED) 
please refer to
[how to use docker](tutorials/how_to_use_docker.md).

### 2. Build From Source

#### Install Toolchains
  
- Install cmake

    Install cmake>=3.14.0, you could refer to [cmake website](https://cmake.org/install) for more detailed info.

    ```bash
    sudo apt-get install -y libssl-dev
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
    tar -zxvf cmake-3.20.0.tar.gz
    cd cmake-3.20.0
    ./bootstrap
    make
    sudo make install
    ```

- GCC 7+

    MMDeploy requires compilers that support C++17.
    ```bash
    # Add repository if ubuntu < 18.04
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test

    sudo apt-get install gcc-7
    sudo apt-get install g++-7
    ```

#### Install Dependencies for Model Converter

- Create a conda virtual environment and activate it

    ```bash
    conda create -n mmdeploy python=3.7 -y
    conda activate mmdeploy
    ```

- Install PyTorch>=1.8.0, following the [official instructions](https://pytorch.org/)

    ```bash
    # CUDA 11.1
    conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    ```

- Install mmcv-full. Refer to the [guide](https://github.com/open-mmlab/mmcv#installation) for details.

    ```bash
    export cu_version=cu111 # cuda 11.1
    export torch_version=torch1.8.0
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/${cu_version}/${torch_version}/index.html
    ```

#### Install Dependencies for SDK

Readers can skip this chapter if only interested in model converter.

The following packages are required to build MMDeploy SDK. Each package's installation command is given based on Ubuntu 18.04.

- OpenCV 3+

  ```bash
  sudo apt-get install libopencv-dev
  ```

- spdlog 0.16+

  ``` bash
  sudo apt-get install libspdlog-dev
  ```

  On Ubuntu 16.04, please use the following command
  ```bash
  wget http://archive.ubuntu.com/ubuntu/pool/universe/s/spdlog/libspdlog-dev_0.16.3-1_amd64.deb
  sudo dpkg -i libspdlog-dev_0.16.3-1_amd64.deb
  ```

  You can also build spdlog from its source to enjoy its latest features. But be sure to add **`-fPIC`** compilation flags at first.

- pplcv

  A high-performance image processing library of openPPL supporting x86 and cuda platforms.</br>
  It is **OPTIONAL** which only be needed if `cuda` platform is required.
  ```bash
  git clone git@github.com:openppl-public/ppl.cv.git
  cd ppl.cv
  ./build.sh cuda
  ```

#### Install Inference Engines for MMDeploy

Both MMDeploy's model converter and SDK share the same inference engines.

Users can select their interested inference engines and do the installation by following the command.

##### ONNXRuntime

*Please note that only **onnxruntime>=1.8.1** of CPU version on Linux platform is supported by now.*
```bash
# install ONNXRuntime python package
pip install onnxruntime==1.8.1

# install ONNXRuntime's library for building custom ops
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

Note that if you want to save onnxruntime env variables to bashrc, you could run

    ```bash
    echo '# set env for onnxruntime' >> ~/.bashrc
    echo "export ONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}" >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```


##### TensorRT

Please install TensorRT 8 by following [install-guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing).

**Note**:

- `pip Wheel File Installation` is not supported yet in this repo.
- We strongly suggest you install TensorRT through [tar file](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar)
- After installation, you'd better add TensorRT environment variables to bashrc by:

    ```bash
    cd ${TENSORRT_DIR} # To TensorRT root directory
    echo '# set env for TensorRT' >> ~/.bashrc
    echo "export TENSORRT_DIR=${TENSORRT_DIR}" >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$TENSORRT_DIR' >> ~/.bashrc
    source ~/.bashrc
    ```

##### ncnn

MMDeploy now supports ncnn version == 1.0.20211208
Please refer to [how-to-build](https://github.com/Tencent/ncnn/wiki/how-to-build) to build and install ncnn and pyncnn

##### pplnn

Please install [ppl](https://github.com/openppl-public/ppl.nn) following [install-guide](https://github.com/openppl-public/ppl.nn/blob/master/docs/en/building-from-source.md).

##### OpenVINO

It is recommended to use the installer or install using pip. Installation example using [pip](https://pypi.org/project/openvino-dev/):
```bash
pip install openvino-dev
```

If you want to use OpenVINO in MMDeploy SDK, you need install OpenVINO with [install_guides](https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_linux.html#install-openvino).
   

#### Build Model Converter

If one of inference engines, such as ONNXRuntime, TensorRT and ncnn is selected, users have to build the corresponding custom ops.

1. Build ONNXRuntime Custom Ops
   
  ```bash
  cd ${MMDEPLOY_DIR} # To MMDeploy root directory
  mkdir -p build && cd build
  cmake -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
  make -j$(nproc)
  ```

2. Build TensorRT Custom Ops
   
  ```bash
  cd ${MMDEPLOY_DIR} # To MMDeploy root directory
  mkdir -p build && cd build
  cmake -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} ..
  make -j$(nproc)
  ```

3. Build ncnn Custom Ops
   
```bash
cd ${MMDEPLOY_DIR} # To MMDeploy root directory
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=ncnn -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn ..
make -j$(nproc)
```

4. Install

```bash
cd ${MMDEPLOY_DIR} # 切换至项目根目录
pip install -e .
```
#### Build SDK

- Enabling Devices

   By default, only CPU device is included in the target devices. You can enable device support for other devices by
   passing a semicolon separated list of device names to `MMDEPLOY_TARGET_DEVICES` variable, e.g. `-DMMDEPLOY_TARGET_DEVICES="cpu;cuda"`. </br>
   Currently, the following devices are supported.

   | device | name | path setter                       |
   | ------ | ---- | --------------------------------- |
   | Host   | cpu  | N/A                               |
   | CUDA   | cuda | CUDA_TOOLKIT_ROOT_DIR & pplcv_DIR |

   If you have multiple CUDA versions installed on your system, you will need to pass `CUDA_TOOLKIT_ROOT_DIR` to cmake to specify the version. </br>
   Meanwhile, `pplcv_DIR` has to be provided in order to build image processing operators on cuda platform.


- Enabling inference engines

   **By default, no target inference engines are set**, since it's highly dependent on the use case.
   `MMDEPLOY_TARGET_BACKENDS` must be set to a semicolon separated list of inference engine names,
   e.g. `-DMMDEPLOY_TARGET_BACKENDS="trt;ort;pplnn;ncnn;openvino"`
   A path to the inference engine library is also needed. The following backends are currently supported

   | library     | name     | path setter              |
   | ----------- | -------- | ------------------------ |
   | PPL.nn      | pplnn    | pplnn_DIR                |
   | ncnn        | ncnn     | ncnn_DIR                 |
   | ONNXRuntime | ort      | ONNXRUNTIME_DIR          |
   | TensorRT    | trt      | TENSORRT_DIR & CUDNN_DIR |
   | OpenVINO    | openvino | InferenceEngine_DIR      |

- Enabling codebase's postprocess components

  `MMDEPLOY_CODEBASES` MUST be specified by a semicolon separated list of codebase names.
  The currently supported codebases are 'mmcls', 'mmdet', 'mmedit', 'mmseg', 'mmocr'.
  Instead of listing them one by one in `MMDEPLOY_CODEBASES`, user can also pass `all` to enable all of them, i.e.,
  `-DMMDEPLOY_CODEBASES=all`


- Put it all together

  The following is a recipe for building MMDeploy SDK with cpu device and ONNXRuntime support
  ```Bash
  mkdir build && cd build
  cmake .. \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DCMAKE_CXX_COMPILER=g++-7 \
      -DONNXRUNTIME_DIR=/path/to/onnxruntime \
      -DMMDEPLOY_TARGET_DEVICES=cpu \
      -DMMDEPLOY_TARGET_BACKENDS=ort \
      -DMMDEPLOY_CODEBASES=all
  cmake --build . -- -j$(nproc) && cmake --install .
  ```

  Here is another example to build MMDeploy SDK with cuda device and TensorRT backend

  ```Bash
   mkdir build && cd build
   cmake .. \
     -DMMDEPLOY_BUILD_SDK=ON \
     -DCMAKE_CXX_COMPILER=g++-7 \
     -Dpplcv_DIR=/path/to/ppl.cv/cuda-build/install/lib/cmake/ppl \
     -DTENSORRT_DIR=/path/to/tensorrt \
     -DCUDNN_DIR=/path/to/cudnn \
     -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
     -DMMDEPLOY_TARGET_BACKENDS=trt \
     -DMMDEPLOY_CODEBASES=all
   cmake --build . -- -j$(nproc) && cmake --install .
  ```

### 3. Use Prebuit Package
TODO

## Build for Windows-x86_64

### 1. Build From Source

### 2. Use Prebuit Package
TODO