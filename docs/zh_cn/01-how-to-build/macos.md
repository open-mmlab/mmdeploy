# Macos 下构建方式

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
conda create -n mmdeploy python=3.9 -y
conda activate mmdeploy
</code></pre>
    </td>
  </tr>
  <tr>
    <td>PyTorch <br>(>=1.8.0) </td>
    <td>安装 PyTorch，要求版本是 torch>=1.8.0。可查看<a href="https://pytorch.org/">官网</a>获取更多详细的安装教程。
<pre><code>
conda install pytorch==1.8.0 torchvision==0.9.0 -c pytorch -c conda-forge
</code></pre>
    </td>
  </tr>
  <tr>
    <td>mmcv-full </td>
    <td>参考如下命令安装 mmcv-full。更多安装方式，可查看 <a href="https://github.com/open-mmlab/mmcv">mmcv 官网</a><br>
<pre><code>
export torch_version=torch1.8
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cpu/${torch_version}/index.html
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
<pre><code>
brew install opencv
</code></pre>
    </td>
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
    <td>Core ML</td>
    <td>coremltools</td>
    <td>
<pre><code>
pip install coremltools
</code></pre>
    </td>

</tr>
  <tr>
    <td>ONNXRuntime</td>
    <td>onnxruntime<br>(>=1.10.0) </td>
    <td>
    1. 安装 onnxruntime 的 python 包
       <pre><code>pip install onnxruntime==1.10.0</code></pre>
    2. 从<a href="https://github.com/microsoft/onnxruntime/releases/tag/v1.10.0">这里</a>下载 onnxruntime 的预编译包。参考如下命令，解压压缩包并设置环境变量
<pre><code>
wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-osx-arm64-1.10.0.tgz
tar -zxvf onnxruntime-osx-arm64-1.10.0.tgz
cd onnxruntime-osx-arm64-1.10.0
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
</code></pre>
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
  1. libtorch暂不提供arm版本的library，故需要自行编译。编译时注意libtorch要和pytorch的版本保持一致，这样编译出的自定义算子才可以加载成功。<br>
  2. 以libtorch 1.8.0为例，可通过如下命令安装:
<pre><code>
git clone -b v1.8.0 --recursive https://github.com/pytorch/pytorch.git
cd pytorch
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=`which python` \
    -DCMAKE_INSTALL_PREFIX=install \
    -DDISABLE_SVE=ON
make -j$(nproc) && make install
export Torch_DIR=$(pwd)/install/share/cmake/Torch
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

如果您选择了ONNXRuntime，ncnn， 和 torchscript 任一种推理后端，您需要编译对应的自定义算子库。

- **Core ML**

  Core ML使用torchscript作为IR，故需要编译torchscript自定义算子。

- **ONNXRuntime** 自定义算子

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
  make -j$(nproc) && make install
  ```

- **ncnn** 自定义算子

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DMMDEPLOY_TARGET_BACKENDS=ncnn -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn ..
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

#### 编译 SDK 和 Demos

下文展示2个构建SDK的样例，分别用 ONNXRuntime 和 Core ML 作为推理引擎。您可以参考它们，激活其他的推理引擎。

- cpu + ONNXRuntime

  ```Bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake .. \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_EXAMPLES=ON \
      -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
      -DMMDEPLOY_TARGET_DEVICES=cpu \
      -DMMDEPLOY_TARGET_BACKENDS=ort \
      -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}

  make -j$(nproc) && make install
  ```

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

  make -j$(nproc) && make install
  ```
