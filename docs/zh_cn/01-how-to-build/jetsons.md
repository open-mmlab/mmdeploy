# 如何在 Jetson 模组上安装 MMDeploy

本教程将介绍如何在 NVIDIA Jetson 平台上安装 MMDeploy。该方法已经在以下 3 种 Jetson 模组上进行了验证：

- Jetson Nano
- Jetson TX2
- Jetson AGX Xavier

## 预备

首先需要在 Jetson 模组上安装 JetPack SDK。
此外，在利用 MMDeploy 的 Model Converter 转换 PyTorch 模型为 ONNX 模型时，需要创建一个装有 PyTorch 的环境。
最后，关于编译工具链，要求 CMake 和 GCC 的版本分别不低于 3.14 和 7.0。

### JetPack SDK

JetPack SDK 为构建硬件加速的边缘 AI 应用提供了一个全面的开发环境。
其支持所有的 Jetson 模组及开发套件。

主要有两种安装 JetPack SDK 的方式：

1. 使用 SD 卡镜像方式，直接将镜像刻录到 SD 卡上
2. 使用 NVIDIA SDK Manager 进行安装

你可以在 NVIDIA [官网](https://developer.nvidia.com/jetpack-sdk-50dp)上找到详细的安装指南。

这里我们选择 [JetPack 4.6.1](https://developer.nvidia.com/jetpack-sdk-461) 作为装配 Jetson 模组的首选。MMDeploy 已经在 JetPack 4.6 rev3 及以上版本，TensorRT 8.0.1.6 及以上版本进行了测试。更早的 JetPack 版本与 TensorRT 7.x 存在不兼容的情况。

### Conda

安装 [Archiconda](https://github.com/Archiconda/build-tools/releases) 而不是 Anaconda，因为后者不提供针对 Jetson 的 wheel 文件。

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

完成安装后需创建并启动一个 conda 环境。

```shell
# 得到默认安装的 python3 版本
export PYTHON_VERSION=`python3 --version | cut -d' ' -f 2 | cut -d'.' -f1,2`
conda create -y -n mmdeploy python=${PYTHON_VERSION}
conda activate mmdeploy
```

```{note}
JetPack SDK 4+ 自带 python 3.6。我们强烈建议使用默认的 python 版本。尝试升级 python 可能会破坏 JetPack 环境。

如果必须安装更高版本的 python， 可以选择安装 JetPack 5+，其提供 python 3.8。
```

### PyTorch

从[这里](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)下载 Jetson 的 PyTorch wheel 文件并保存在本地目录 `/opt` 中。
此外，由于 torchvision 不提供针对 Jetson 平台的预编译包，因此需要从源码进行编译。

以 `torch 1.10.0` 和  `torchvision 0.11.1` 为例，可按以下方式进行安装：

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

如果安装其他版本的 PyTorch 和 torchvision，需参考[这里](https://pypi.org/project/torchvision/)的表格以保证版本兼容性。

### CMake

这里我们使用 CMake 截至2022年4月的最新版本 v3.23.1。

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

## 安装依赖项

MMDeploy 中的 Model Converter 依赖于 [MMCV](https://github.com/open-mmlab/mmcv) 和推理引擎 [TensorRT](https://developer.nvidia.com/tensorrt)。
同时， MMDeploy 的 C/C++ Inference SDK 依赖于 [spdlog](https://github.com/gabime/spdlog)， OpenCV， [ppl.cv](https://github.com/openppl-public/ppl.cv) 和 TensorRT 等。
因此，接下来我们将先介绍如何配置 TensorRT。
之后再分别展示安装 Model Converter 和 C/C++ Inference SDK 的步骤。

### 配置 TensorRT

JetPack SDK 自带 TensorRT。
但是为了能够在 Conda 环境中成功导入，我们需要将 TensorRT 拷贝进先前创建的 Conda 环境中。

```shell
cp -r /usr/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* ~/archiconda3/envs/mmdeploy/lib/python${PYTHON_VERSION}/site-packages/
conda deactivate
conda activate mmdeploy
python -c "import tensorrt; print(tensorrt.__version__)" # 将会打印出 TensorRT 版本

# 为之后编译 MMDeploy 设置环境变量
export TENSORRT_DIR=/usr/include/aarch64-linux-gnu

# 将 cuda 路径和 lib 路径写入到环境变量 `$PATH` 和 `$LD_LIBRARY_PATH` 中， 为之后编译 MMDeploy 做准备
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

你也可以通过添加以上环境变量至 `~/.bashrc` 使得它们永久化。

```shell
echo -e '\n# set environment variable for TensorRT' >> ~/.bashrc
echo 'export TENSORRT_DIR=/usr/include/aarch64-linux-gnu' >> ~/.bashrc

echo -e '\n# set environment variable for CUDA' >> ~/.bashrc
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc

source ~/.bashrc
conda activate mmdeploy
```

### 安装 Model Converter 的依赖项

- 安装 [MMCV](https://github.com/open-mmlab/mmcv)

  MMCV 还未提供针对 Jetson 平台的预编译包，因此我们需要从源对其进行编译。

  ```shell
  sudo apt-get install -y libssl-dev
  git clone https://github.com/open-mmlab/mmcv.git
  cd mmcv
  git checkout v1.4.0
  MMCV_WITH_OPS=1 pip install -e .
  ```

- 安装 ONNX

  ```shell
  pip install onnx
  ```

- 安装 h5py

  Model Converter 使用 HDF5 存储 TensorRT INT8 量化的校准数据。

  ```shell
  sudo apt-get install -y pkg-config libhdf5-100 libhdf5-dev
  pip install versioned-hdf5
  ```

### 安装 SDK 的依赖项

如果你不需要使用 MMDeploy C/C++ Inference SDK 则可以跳过本步骤。

- 安装 [spdlog](https://github.com/gabime/spdlog)

  “`spdlog` 是一个快速的，仅有头文件的 C++ 日志库。”

  ```shell
  sudo apt-get install -y libspdlog-dev
  ```

- 安装 [ppl.cv](https://github.com/openppl-public/ppl.cv)

  “`ppl.cv` 是 [OpenPPL](https://openppl.ai/home) 的高性能图像处理库。”

  ```shell
  git clone https://github.com/openppl-public/ppl.cv.git
  cd ppl.cv
  export PPLCV_DIR=$(pwd)
  echo -e '\n# set environment variable for ppl.cv' >> ~/.bashrc
  echo "export PPLCV_DIR=$(pwd)" >> ~/.bashrc
  ./build.sh cuda
  ```

## 安装 MMDeploy

```shell
git clone --recursive https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
export MMDEPLOY_DIR=$(pwd)
```

### 安装 Model Converter

由于一些算子采用的是 OpenMMLab 代码库中的实现，并不被 TenorRT 支持，
因此我们需要自定义 TensorRT 插件，例如 `roi_align`， `scatternd` 等。
你可以从[这里](../06-custom-ops/tensorrt.md)找到完整的自定义插件列表。

```shell
# 编译 TensorRT 自定义算子
mkdir -p build && cd build
cmake .. -DMMDEPLOY_TARGET_BACKENDS="trt"
make -j$(nproc) && make install

# 安装 model converter
cd ${MMDEPLOY_DIR}
pip install -v -e .
# "-v" 表示显示详细安装信息
# "-e" 表示在可编辑模式下安装
# 因此任何针对代码的本地修改都可以在无需重装的情况下生效。
```

### 安装 C/C++ Inference SDK

如果你不需要使用 MMDeploy C/C++ Inference SDK 则可以跳过本步骤。

1. 编译 SDK Libraries 和 Demos

   ```shell
   mkdir -p build && cd build
   cmake .. \
       -DMMDEPLOY_BUILD_SDK=ON \
       -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
       -DMMDEPLOY_BUILD_EXAMPLES=ON \
       -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
       -DMMDEPLOY_TARGET_BACKENDS="trt" \
       -DMMDEPLOY_CODEBASES=all \
       -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl
   make -j$(nproc) && make install
   ```

2. 运行 demo

   以目标检测为例:

   ```shell
   ./object_detection cuda ${directory/to/the/converted/models} ${path/to/an/image}
   ```

## Troubleshooting

### 安装

- `pip install` 报错 `Illegal instruction (core dumped)`

  ```shell
  echo '# set env for pip' >> ~/.bashrc
  echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
  source ~/.bashrc
  ```

  如果上述方法仍无法解决问题，检查是否正在使用镜像文件。如果是的，可尝试：

  ```shell
  rm .condarc
  conda clean -i
  conda create -n xxx python=${PYTHON_VERSION}
  ```

### 执行

- `#assertion/root/workspace/mmdeploy/csrc/backend_ops/tensorrt/batched_nms/trt_batched_nms.cpp,98` or `pre_top_k need to be reduced for devices with arch 7.2`

  1. 设置为 `MAX N` 模式并执行 `sudo nvpmodel -m 0 && sudo jetson_clocks`。
  2. 效仿 [mmdet pre_top_k](https://github.com/open-mmlab/mmdeploy/blob/34879e638cc2db511e798a376b9a4b9932660fe1/configs/mmdet/_base_/base_static.py#L13)，减少配置文件中 `pre_top_k` 的个数，例如 `1000`。
  3. 重新进行模型转换并重新运行 demo。
