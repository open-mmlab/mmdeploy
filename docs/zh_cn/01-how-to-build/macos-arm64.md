# macOS-arm64 下构建方式

- [macOS-arm64 下构建方式](#macos-arm64-下构建方式)
  - [源码安装](#源码安装)
    - [安装构建和编译工具链](#安装构建和编译工具链)
    - [安装依赖包](#安装依赖包)
      - [安装 MMDeploy Converter 依赖](#安装-mmdeploy-converter-依赖)
      - [安装 MMDeploy SDK 依赖](#安装-mmdeploy-sdk-依赖)
      - [安装推理引擎](#安装推理引擎)
    - [编译 MMDeploy](#编译-mmdeploy)
      - [编译 Model Converter](#编译-model-converter)
      - [安装 Model Converter](#安装-model-converter)
      - [编译 SDK 和 Demos](#编译-sdk-和-demos)

## 源码安装

### 安装构建和编译工具链

- cmake

  ```
  brew install cmake
  ```

- clang

  安装 Xcode 或者通过如下命令安装 Command Line Tools

  ```
  xcode-select --install
  ```

### 安装依赖包

#### 安装 MMDeploy Converter 依赖

参考[get_started](../get_started.md)文档，安装conda。

```bash
# install pytoch & mmcv-full
conda install pytorch==1.9.0 torchvision==0.10.0 -c pytorch
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html
```

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
<pre><code>
brew install opencv
</code></pre>
    </td>
</tbody>
</table>

#### 安装推理引擎

MMDeploy 的 Model Converter 和 SDK 共享推理引擎。您可以参考下文，选择自己感兴趣的推理引擎安装。这里重点介绍 Core ML。ONNX Runtime，ncnn 以及 TorchScript 的安装类似 linux 平台，可参考文档 [linux-x86_64](linux-x86_64.md) 进行安装。

Core ML 模型的转化过程中使用 TorchScript 模型作为IR，为了支持含有自定义算子的模型，如 mmdet 中的检测模型，需要安装 libtorch，这里作简单说明。

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
    <td>Core ML</td>
    <td>coremltools</td>
    <td>
<pre><code>
pip install coremltools==6.0b2
</code></pre>
    </td>
  </tr>
  <tr>
  <td>TorchScript</td>
  <td>libtorch</td>
  <td>
  1. libtorch暂不提供arm版本的library，故需要自行编译。编译时注意libtorch要和pytorch的版本保持一致，这样编译出的自定义算子才可以加载成功。<br>
  2. 以libtorch 1.9.0为例，可通过如下命令安装:
<pre><code>
git clone -b v1.9.0 --recursive https://github.com/pytorch/pytorch.git
cd pytorch
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=`which python` \
    -DCMAKE_INSTALL_PREFIX=install \
    -DDISABLE_SVE=ON # low version like 1.9.0 of pytorch need DISABLE_SVE option
make -j4 && make install
export Torch_DIR=$(pwd)/install/share/cmake/Torch
</code></pre>
  </td>
  </tr>
</tbody>
</table>

### 编译 MMDeploy

```bash
cd /the/root/path/of/MMDeploy
export MMDEPLOY_DIR=$(pwd)
```

#### 编译 Model Converter

这里介绍使用 Core ML 作为推理后端所需的操作。

- **Core ML**

  Core ML使用 torchscript 作为IR，某些 codebase 如 mmdet 需要编译 torchscript 自定义算子。

- **torchscript** 自定义算子

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DMMDEPLOY_TARGET_BACKENDS=coreml -DTorch_DIR=${Torch_DIR} ..
  make -j4 && make install
  ```

参考 [cmake 选项说明](cmake_option.md)

#### 安装 Model Converter

```bash
# requirements/runtime.txt 中依赖项grpcio，通过pip安装的方式无法正常import, 需使用 conda 安装
conda install grpcio
```

```bash
cd ${MMDEPLOY_DIR}
pip install -v -e .
```

**注意**

- 有些依赖项是可选的。运行 `pip install -e .` 将进行最小化依赖安装。 如果需安装其他可选依赖项，请执行`pip install -r requirements/optional.txt`，
  或者 `pip install -e .[optional]`。其中，`[optional]`可以替换为：`all`、`tests`、`build` 或 `optional`。

#### 编译 SDK 和 Demos

下文展示使用 Core ML 作为推理引擎，构建SDK的样例。

- cpu + Core ML

  ```Bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake .. \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_EXAMPLES=ON \
      -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
      -DMMDEPLOY_TARGET_DEVICES=cpu \
      -DMMDEPLOY_TARGET_BACKENDS=coreml \
      -DTorch_DIR=${Torch_DIR}

  make -j4 && make install
  ```
