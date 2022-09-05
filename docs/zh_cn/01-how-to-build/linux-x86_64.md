# Linux-x86_64 下构建方式

- [Linux-x86_64 下构建方式](#linux-x86_64-下构建方式)
  - [源码安装](#源码安装)
    - [安装构建和编译工具链](#安装构建和编译工具链)
    - [安装依赖包](#安装依赖包)
      - [安装 MMDeploy Converter 依赖](#安装-mmdeploy-converter-依赖)
      - [安装 MMDeploy SDK 依赖](#安装-mmdeploy-sdk-依赖)
      - [安装推理引擎](#安装推理引擎)
    - [编译 MMDeploy](#编译-mmdeploy)
      - [编译选项说明](#编译选项说明)
      - [编译安装 Model Converter](#编译安装-model-converter)
        - [编译自定义算子](#编译自定义算子)
        - [安装 Model Converter](#安装-model-converter)
      - [编译 SDK 和 Demos](#编译-sdk-和-demos)

______________________________________________________________________

## 源码安装

### 安装构建和编译工具链

- cmake

  **保证 cmake的版本 >= 3.14.0**。如果不是，可以参考以下命令安装 3.20.0 版本。如需获取其他版本，请参考[这里](https://cmake.org/install)。

  ```bash
  wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.tar.gz
  tar -xzvf cmake-3.20.0-linux-x86_64.tar.gz
  sudo ln -sf $(pwd)/cmake-3.20.0-linux-x86_64/bin/* /usr/bin/
  ```

- GCC 7+

  MMDeploy SDK 使用了 C++17 特性，因此需要安装gcc 7+以上的版本。

  ```bash
  # 如果 Ubuntu 版本 < 18.04，需要加入仓库
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test
  sudo apt-get update
  sudo apt-get install gcc-7
  sudo apt-get install g++-7
  ```

### 安装依赖包

#### 安装 MMDeploy Converter 依赖

<table class="docutils">
<thead>
  <tr>
    <th>名称 </th>
    <th>安装说明 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>conda </td>
    <td>请参考<a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">官方说明</a>安装 conda。<br> 通过 conda 创建并激活 Python 环境。<br>
<pre><code>
conda create -n mmdeploy python=3.7 -y
conda activate mmdeploy
</code></pre>
    </td>
  </tr>
  <tr>
    <td>PyTorch <br>(>=1.8.0) </td>
    <td>安装 PyTorch，要求版本是 torch>=1.8.0。可查看<a href="https://pytorch.org/">官网</a>获取更多详细的安装教程。请确保 PyTorch 要求的 CUDA 版本和您主机的 CUDA 版本是一致<br>
<pre><code>
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
</code></pre>
    </td>
  </tr>
  <tr>
    <td>mmcv-full </td>
    <td>参考如下命令安装 mmcv-full。更多安装方式，可查看 <a href="https://github.com/open-mmlab/mmcv">mmcv 官网</a><br>
<pre><code>
export cu_version=cu111 # cuda 11.1
export torch_version=torch1.8
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/${cu_version}/${torch_version}/index.html
</code></pre>
    </td>
  </tr>
</tbody>
</table>

#### 安装 MMDeploy SDK 依赖

如果您只对模型转换感兴趣，那么可以跳过本章节。

<table class="docutils">
<thead>
  <tr>
    <th>名称 </th>
    <th>安装说明 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>OpenCV<br>(>=3.0) </td>
    <td>
    在 Ubuntu 18.04 及以上版本
<pre><code>
sudo apt-get install libopencv-dev
</code></pre>
    在 Ubuntu 16.04 中，需要源码安装 OpenCV。您可以参考<a href="https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html">此处</a>.
    </td>

</tr>
  <tr>
    <td>pplcv </td>
    <td>pplcv 是 openPPL 开发的高性能图像处理库。 <b>此依赖项为可选项，只有在 cuda 平台下，才需安装。</b><br>
<pre><code>
git clone https://github.com/openppl-public/ppl.cv.git
cd ppl.cv
export PPLCV_DIR=$(pwd)
git checkout tags/v0.7.0 -b v0.7.0
./build.sh cuda
</code></pre>
   </td>
  </tr>
</tbody>
</table>

#### 安装推理引擎

MMDeploy 的 Model Converter 和 SDK 共享推理引擎。您可以参考下文，选择自己感兴趣的推理引擎安装。

<table  class="docutils">
<thead>
  <tr>
    <th>名称</th>
    <th>安装包</th>
    <th>安装说明</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ONNXRuntime</td>
    <td>onnxruntime<br>(>=1.8.1) </td>
    <td>
    1. 安装 onnxruntime 的 python 包
       <pre><code>pip install onnxruntime==1.8.1</code></pre>
    2. 从<a href="https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1">这里</a>下载 onnxruntime 的预编译包。参考如下命令，解压压缩包并设置环境变量
<pre><code>
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
</code></pre>
    </td>
  </tr>
  <tr>
    <td rowspan="2">TensorRT<br> </td>
    <td>TensorRT <br> </td>
    <td>
   1. 登录 <a href="https://www.nvidia.com/">NVIDIA 官网</a>，从<a href="https://developer.nvidia.com/nvidia-tensorrt-download">这里</a>选取并下载 TensorRT tar 包。要保证它和您机器的 CPU 架构以及 CUDA 版本是匹配的。<br>
   您可以参考这份<a href="https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar">指南</a>安装 TensorRT。<br>
   1. 这里也有一份 TensorRT 8.2 GA Update 2 在 Linux x86_64 和 CUDA 11.x 下的安装示例，供您参考。首先，点击<a href="https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.3.0/tars/tensorrt-8.2.3.0.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz">此处</a>下载 CUDA 11.x TensorRT 8.2.3.0。然后，根据如下命令，安装并配置 TensorRT 以及相关依赖。
<pre><code>
cd /the/path/of/tensorrt/tar/gz/file
tar -zxvf TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
pip install TensorRT-8.2.3.0/python/tensorrt-8.2.3.0-cp37-none-linux_x86_64.whl
export TENSORRT_DIR=$(pwd)/TensorRT-8.2.3.0
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
pip install pycuda
</code></pre>
   </td>
  </tr>
  <tr>
    <td>cuDNN </td>
    <td>
    1. 从 <a href="https://developer.nvidia.com/rdp/cudnn-archive">cuDNN Archive</a> 选择和您环境中 CPU 架构、CUDA 版本以及 TensorRT 版本配套的 cuDNN。以前文 TensorRT 安装说明为例，它需要 cudnn8.2。因此，可以下载 <a href="https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-linux-x64-v8.2.1.32.tgz">CUDA 11.x cuDNN 8.2</a><br>
    2. 解压压缩包，并设置环境变量
<pre><code>
cd /the/path/of/cudnn/tgz/file
tar -zxvf cudnn-11.3-linux-x64-v8.2.1.32.tgz
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
</code></pre>
   </td>
  </tr>
  <tr>
    <td>PPL.NN</td>
    <td>ppl.nn </td>
    <td>
    1. 请参考 ppl.nn 的 <a href="https://github.com/openppl-public/ppl.nn/blob/master/docs/en/building-from-source.md">安装文档</a> 编译 ppl.nn，并安装 pyppl<br>
    2. 将 pplnn 的根目录写入环境变量
<pre><code>
cd ppl.nn
export PPLNN_DIR=$(pwd)
</code></pre>
    </td>
  </tr>
  <tr>
    <td>OpenVINO</td>
    <td>openvino </td>
    <td>1. 安装 <a href="https://docs.openvino.ai/2021.4/get_started.html">OpenVINO</a>
<pre><code>
pip install openvino-dev
</code></pre>
2. <b>可选</b>. 如果您想在 MMDeploy SDK 中使用 OpenVINO，请根据<a href="https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_linux.html#install-openvino">指南</a>安装并配置它
    </td>
  </tr>
  <tr>
    <td>ncnn </td>
    <td>ncnn </td>
    <td>1. 请参考 ncnn的 <a href="https://github.com/Tencent/ncnn/wiki/how-to-build">wiki</a> 编译 ncnn。
编译时，请打开<code>-DNCNN_PYTHON=ON</code><br>
2. 将 ncnn 的根目录写入环境变量
<pre><code>
cd ncnn
export NCNN_DIR=$(pwd)
</code></pre>
3. 安装 pyncnn
<pre><code>
cd ${NCNN_DIR}/python
pip install -e .
</code></pre>
    </td>
  </tr>
  <tr>
  <td>TorchScript</td>
  <td>libtorch</td>
  <td>
  1. Download libtorch from <a href="https://pytorch.org/get-started/locally/">here</a>. Please note that only <b>Pre-cxx11 ABI</b> and <b>version 1.8.1+</b> on Linux platform are supported by now. For previous versions of libtorch, you can find them in the <a href="https://github.com/pytorch/pytorch/issues/40961#issuecomment-1017317786">issue comment</a>. <br>
  2. Take Libtorch1.8.1+cu111 as an example. You can install it like this:
  <pre><code>
wget https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.8.1%2Bcu111.zip
unzip libtorch-shared-with-deps-1.8.1+cu111.zip
cd libtorch
export Torch_DIR=$(pwd)
export LD_LIBRARY_PATH=$Torch_DIR/lib:$LD_LIBRARY_PATH
  </code></pre>
  </td>
  </tr>
  <tr>
    <td>Ascend</td>
    <td>CANN</td>
    <td>
    1. 按照 <a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/60RC1alpha02/softwareinstall/instg/atlasdeploy_03_0002.html">官方指引</a> 安装 CANN 工具集.<br>
    2. 配置环境
   <pre><code>
export ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
   </code></pre>
    </td>
  </tr>
</tbody>
</table>

注意: <br>
如果您想使上述环境变量永久有效，可以把它们加入<code>~/.bashrc</code>。以 ONNXRuntime 的环境变量为例，

```bash
echo '# set env for onnxruntime' >> ~/.bashrc
echo "export ONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

### 编译 MMDeploy

```bash
cd /the/root/path/of/MMDeploy
export MMDEPLOY_DIR=$(pwd)
```

#### 编译 Model Converter

如果您选择了ONNXRuntime，TensorRT，ncnn 和 torchscript 任一种推理后端，您需要编译对应的自定义算子库。

- **ONNXRuntime** 自定义算子

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
  make -j$(nproc) && make install
  ```

- **TensorRT** 自定义算子

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} -DCUDNN_DIR=${CUDNN_DIR} ..
  make -j$(nproc) && make install
  ```

- **ncnn** 自定义算子

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=ncnn -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn ..
  make -j$(nproc) && make install
  ```

- **torchscript** 自定义算子

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DMMDEPLOY_TARGET_BACKENDS=torchscript -DTorch_DIR=${Torch_DIR} ..
  make -j$(nproc) && make install
  ```

参考 [cmake 选项说明](cmake_option.md)

#### 安装 Model Converter

```bash
cd ${MMDEPLOY_DIR}
pip install -e .
```

**注意**

- 有些依赖项是可选的。运行 `pip install -e .` 将进行最小化依赖安装。 如果需安装其他可选依赖项，请执行`pip install -r requirements/optional.txt`，
  或者 `pip install -e .[optional]`。其中，`[optional]`可以替换为：`all`、`tests`、`build` 或 `optional`。
- cuda10 建议安装[补丁包](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)，否则模型运行可能出现 GEMM 相关错误

#### 编译 SDK 和 Demos

下文展示2个构建SDK的样例，分别用 ONNXRuntime 和 TensorRT 作为推理引擎。您可以参考它们，激活其他的推理引擎。

- cpu + ONNXRuntime

  ```Bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake .. \
      -DCMAKE_CXX_COMPILER=g++-7 \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_EXAMPLES=ON \
      -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
      -DMMDEPLOY_TARGET_DEVICES=cpu \
      -DMMDEPLOY_TARGET_BACKENDS=ort \
      -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}

  make -j$(nproc) && make install
  ```

- cuda + TensorRT

  ```Bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake .. \
      -DCMAKE_CXX_COMPILER=g++-7 \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_EXAMPLES=ON \
      -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
      -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
      -DMMDEPLOY_TARGET_BACKENDS=trt \
      -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl \
      -DTENSORRT_DIR=${TENSORRT_DIR} \
      -DCUDNN_DIR=${CUDNN_DIR}

  make -j$(nproc) && make install
  ```

- pplnn

  ```Bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake .. \
      -DCMAKE_CXX_COMPILER=g++-7 \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_EXAMPLES=ON \
      -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
      -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
      -DMMDEPLOY_TARGET_BACKENDS=pplnn \
      -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/cuda-build/install/lib/cmake/ppl \
      -Dpplnn_DIR=${PPLNN_DIR}/pplnn-build/install/lib/cmake/ppl

  make -j$(nproc) && make install
  ```
