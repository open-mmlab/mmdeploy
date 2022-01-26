## 安装 MMdeploy

我们提供物理机和虚拟机构建方法。虚拟机搭建方法请参考[如何使用docker](tutorials/how_to_use_docker.md)。对于物理机，请按照以下步骤操作

### 准备工作

- 下载代码仓库 MMDeploy

    ```bash
    git clone -b master git@github.com:open-mmlab/mmdeploy.git MMDeploy
    cd MMDeploy
    export MMDEPLOY_DIR=$(pwd)
    git submodule update --init --recursive
    ```

    提示:

  - 如果由于网络等原因导致拉取仓库子模块失败，可以尝试通过如下指令手动再次安装子模块:

      ```bash
      git clone git@github.com:NVIDIA/cub.git third_party/cub
      cd third_party/cub
      git checkout c3cceac115

      # 返回至 third_party 目录, 克隆 pybind11
      cd ..
      git clone git@github.com:pybind/pybind11.git pybind11
      cd pybind11
      git checkout 70a58c5
      ```

- 安装编译工具 cmake

    要求 cmake>=3.14.0, 通过如下指令安装 cmake。您也通过 [cmake](https://cmake.org/install) 官网查看更多安装信息。

    ```bash
    apt-get install -y libssl-dev
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
    tar -zxvf cmake-3.20.0.tar.gz
    cd cmake-3.20.0
    ./bootstrap
    make
    make install
    ```

- 安装 GCC 7+

  MMDeploy SDK 使用了 C++17 特性，因此需要安装gcc 7+以上的版本。

  ```bash
    # Add repository if ubuntu < 18.04
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt-get install gcc-7
    sudo apt-get install g++-7
    ```

### 创建环境

- 通过 conda 创建并激活 Python 环境

    ```bash
    conda create -n mmdeploy python=3.7 -y
    conda activate mmdeploy
    ```

- 安装 PyTorch，要求版本是 torch>=1.8.0, 可查看[官网](https://pytorch.org/)获取更详细的安装教程。

    ```bash
    # CUDA 11.1
    conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    ```

- 安装 mmcv-full, 更多安装方式可查看[教程](https://github.com/open-mmlab/mmcv#installation)

    ```bash
    export cu_version=cu111 # cuda 11.1
    export torch_version=torch1.8.0
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/${cu_version}/${torch_version}/index.html
    ```

### 安装推理引擎

您可以根据自身需求，构建和安装如下推理引擎：

- [ONNX Runtime](https://mmdeploy.readthedocs.io/en/latest/backends/onnxruntime.html)
- [TensorRT](https://mmdeploy.readthedocs.io/en/latest/backends/tensorrt.html)
- [ncnn](https://mmdeploy.readthedocs.io/en/latest/backends/ncnn.html)
- [PPLNN](https://mmdeploy.readthedocs.io/en/latest/backends/pplnn.html)
- [OpenVINO](https://mmdeploy.readthedocs.io/en/latest/backends/openvino.html)

### 安装 MMDeploy

```bash
cd ${MMDEPLOY_DIR} # 切换至项目根目录
pip install -e .
```

**Note**

- 有些依赖项是可选的。运行 `pip install -e .` 将进行最小化依赖安装。 如果需安装其他可选依赖项，请执行`pip install -r requirements/optional.txt`，
或者 `pip install -e . [optional]`。其中，`[optional]`可以填写`all`, `tests`, `build`, `optional`

### 构建 SDK

读者如果只对模型转换感兴趣，那么可以跳过本章节

#### 安装依赖项

目前，SDK在Linux-x86_64经过测试验证，未来将加入对更多平台的支持。 使用SDK，需要安装若干依赖包。本文以 Ubuntu 18.04为例，逐一介绍各依赖项的安装方法

- OpenCV 3+

  ```bash
  sudo apt-get install libopencv-dev
  ```

- spdlog 0.16+

  ``` bash
  sudo apt-get install libspdlog-dev
  ```

  如果使用 Ubuntu 16.04, 请用如下命令下载并安装合适的spdlog版本

  ```bash
  wget http://archive.ubuntu.com/ubuntu/pool/universe/s/spdlog/libspdlog-dev_0.16.3-1_amd64.deb
  sudo dpkg -i libspdlog-dev_0.16.3-1_amd64.deb
  ```

  你也可以使用spdlog源码编译，激活它更多的特性。但是，请务必打开 **`-fPIC`** 编译选项。

- pplcv

  pplcv 是在x86和cuda平台下的高性能图像处理库。
  此依赖项为可选项，只有在cuda平台下，才需安装。安装命令如下所示:

  ```bash
  git clone git@github.com:openppl-public/ppl.cv.git
  cd ppl.cv
  ./build.sh cuda
  ```

- 推理引擎
  SDK 和 model converter 使用相同的推理引擎。 请参考前文中”安装推理引擎“章节，选择合适的进行安装.

#### 设置编译选项

- 打开 SDK 编译开关

  `-DMMDEPLOY_BUILD_SDK=ON`

- 设置目标设备

  cpu 是 SDK 目标设备的默认选项。你也可以通过`MMDEPLOY_TARGET_DEVICES`传入其他设备名称。当有多个设备时，设备名称之间使用分号隔开。
  比如，`-DMMDEPLOY_TARGET_DEVICES="cpu;cuda"`。
  当前，SDK支持以下设备，

  | 设备 | 名称 | 查找路径                           |
  | :--- | :--- | :--------------------------------- |
  | Host | cpu  | N/A                                |
  | CUDA | cuda | CUDA_TOOLKIT_ROOT_DIR 和 pplcv_DIR |

  如果你的开发环境中有多个cuda版本，则需要通过`-DCUDA_TOOLKIT_ROOT_DIR=/path/of/cuda`来明确使用的版本。
  于此同时，还需设置`-Dpplcv_DIR=ppl.cv/path/install/lib/cmake/ppl`，用以编译cuda平台下的图像处理算子。

- 设置推理后端

  **默认情况下，SDK不设置任何后端**, 因为它与应用场景高度相关。你可以通过设置`MMDEPLOY_TARGET_BACKENDS`激活感兴趣的推理后端。
  当选择多个时， 中间使用分号隔开。比如，`-DMMDEPLOY_TARGET_BACKENDS="trt;ort;pplnn;ncnn;openvino"`
  构建时，几乎每个后端，都需设置一些环境变量，用来查找依赖包。
  下表展示了目前SDK支持的后端，以及构建时，每个后端需要设置的变量。

  | 推理引擎    | 名称     | 查找路径                 |
  | :---------- | :------- | :----------------------- |
  | PPL.nn      | pplnn    | pplnn_DIR                |
  | ncnn        | ncnn     | ncnn_DIR                 |
  | ONNXRuntime | ort      | ONNXRUNTIME_DIR          |
  | TensorRT    | trt      | TENSORRT_DIR & CUDNN_DIR |
  | OpenVINO    | openvino | InferenceEngine_DIR      |

- 设置后处理组件

  需要通过`MMDEPLOY_CODEBASES`设置SDK后处理组件，才能加载OpenMMLab算法仓库的后处理功能。已支持的算法仓库有'mmcls'，'mmdet'，'mmedit'，'mmseg'和'mmocr'。
  如果选择多个codebase，中间使用分号隔开。比如，`-DMMDEPLOY_CODEBASES=mmcls;mmdet`。也可以通过`-DMMDEPLOY_CODEBASES=all`方式，加载所有codebase。

- 汇总以上

  下文展示2个构建SDK的样例，分别用于不同的运行环境。
  使用cpu设备和ONNXRuntime推理，请参考

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

  使用cuda设备和TensorRT推理，请按照此例构建

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
