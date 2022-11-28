# Win10 下构建方式

- [Win10 下构建方式](#win10-下构建方式)
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
    - [注意事项](#注意事项)

______________________________________________________________________

## 源码安装

下述安装方式，均是在 **Windows 10** 下进行，使用 **PowerShell Preview** 版本。

### 安装构建和编译工具链

1. 下载并安装 [Visual Studio 2019](https://visualstudio.microsoft.com) 。安装时请勾选 "使用C++的桌面开发, "Windows 10 SDK <br>
2. 把 cmake 路径加入到环境变量 PATH 中, "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin" <br>
3. 如果系统中配置了 NVIDIA 显卡，根据[官网教程](https://developer.nvidia.com\/cuda-downloads)，下载并安装 cuda toolkit。<br>

### 安装依赖包

#### 安装 MMDeploy Converter 依赖

<table class="docutils">
<thead>
  <tr>
    <th>名称 </th>
    <th>安装方法 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>conda </td>
    <td>请参考 <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">这里</a> 安装 conda。安装完毕后，打开系统开始菜单，<b>以管理员的身份打开 anaconda powershell prompt</b>。 因为，<br>
<b>1. 下文中的安装命令均是在 anaconda powershell 中测试验证的。</b><br>
<b>2. 使用管理员权限，可以把第三方库安装到系统目录。能够简化 MMDeploy 编译命令。</b><br>
<b>说明：如果你对 cmake 工作原理很熟悉，也可以使用普通用户权限打开 anaconda powershell prompt</b>。
</td>
  </tr>
  <tr>
    <td>PyTorch <br>(>=1.8.0) </td>
    <td> 安装 PyTorch，要求版本是 torch>=1.8.0。可查看<a href="https://pytorch.org/">官网</a>获取更多详细的安装教程。请确保 PyTorch 要求的 CUDA 版本和您主机的 CUDA 版本是一致<br>
<pre><code>
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
</code></pre>
    </td>
  </tr>
  <tr>
    <td>mmcv-full </td>
    <td>参考如下命令安装 mmcv-full。更多安装方式，可查看 <a href="https://github.com/open-mmlab/mmcv">mmcv 官网</a><br>
<pre><code>
$env:cu_version="cu111"
$env:torch_version="torch1.8"
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/$env:cu_version/$env:torch_version/index.html
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
    <th>安装方法 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>OpenCV </td>
    <td>
    1. 从<a href="https://github.com/opencv/opencv/releases">这里</a>下载 OpenCV 3+。
    2. 您可以下载并安装 OpenCV 预编译包到指定的目录下。也可以选择源码编译安装的方式
    3. 在安装目录中，找到 <code>OpenCVConfig.cmake</code>，并把它的路径添加到环境变量 <code>PATH</code> 中。像这样：
<pre><code>$env:path = "\the\path\where\OpenCVConfig.cmake\locates;" + "$env:path"</code></pre>
    </td>
  </tr>
  <tr>
    <td>pplcv </td>
    <td>pplcv 是 openPPL 开发的高性能图像处理库。 <b>此依赖项为可选项，只有在 cuda 平台下，才需安装。</b><br>
<pre><code>
git clone https://github.com/openppl-public/ppl.cv.git
cd ppl.cv
git checkout tags/v0.7.0 -b v0.7.0
$env:PPLCV_DIR = "$pwd"
mkdir pplcv-build
cd pplcv-build
cmake .. -G "Visual Studio 16 2019" -T v142 -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DPPLCV_USE_CUDA=ON -DPPLCV_USE_MSVC_STATIC_RUNTIME=OFF
cmake --build . --config Release -- /m
cmake --install . --config Release
cd ../..
</code></pre>
   </td>
  </tr>
</tbody>
</table>

#### 安装推理引擎

MMDeploy 的 Model Converter 和 SDK 共享推理引擎。您可以参考下文，选择自己感兴趣的推理引擎安装。

**目前，在 Windows 平台下，MMDeploy 支持 ONNXRuntime 和 TensorRT 两种推理引擎**。其他推理引擎尚未进行验证，或者验证未通过。后续将陆续予以支持

<table class="docutils">
<thead>
  <tr>
    <th>推理引擎 </th>
    <th>依赖包</th>
    <th>安装方法 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ONNXRuntime</td>
    <td>onnxruntime<br>(>=1.8.1) </td>
    <td>
    1. 安装 onnxruntime 的 python 包
<pre><code>pip install onnxruntime==1.8.1</code></pre>
    2. 从<a href="https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1">这里</a>下载 onnxruntime 的预编译二进制包，解压并配置环境变量
<pre><code>
Invoke-WebRequest -Uri https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-win-x64-1.8.1.zip -OutFile onnxruntime-win-x64-1.8.1.zip
Expand-Archive onnxruntime-win-x64-1.8.1.zip .
$env:ONNXRUNTIME_DIR = "$pwd\onnxruntime-win-x64-1.8.1"
$env:path = "$env:ONNXRUNTIME_DIR\lib;" + $env:path
</code></pre>
    </td>
  </tr>
  <tr>
    <td rowspan="2">TensorRT<br> </td>
    <td>TensorRT <br> </td>
    <td>
    1. 登录 <a href="https://www.nvidia.com/">NVIDIA 官网</a>，从<a href="https://developer.nvidia.com/nvidia-tensorrt-download">这里</a>选取并下载 TensorRT tar 包。要保证它和您机器的 CPU 架构以及 CUDA 版本是匹配的。您可以参考这份 <a href="https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar">指南</a> 安装 TensorRT。<br>
    2. 这里也有一份 TensorRT 8.2 GA Update 2 在 Windows x86_64 和 CUDA 11.x 下的安装示例，供您参考。首先，点击<a href="https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.3.0/zip/TensorRT-8.2.3.0.Windows10.x86_64.cuda-11.4.cudnn8.2.zip">此处</a>下载 CUDA 11.x TensorRT 8.2.3.0。然后，根据如下命令，安装并配置 TensorRT 以及相关依赖。
<pre><code>
cd \the\path\of\tensorrt\zip\file
Expand-Archive TensorRT-8.2.3.0.Windows10.x86_64.cuda-11.4.cudnn8.2.zip .
pip install $env:TENSORRT_DIR\python\tensorrt-8.2.3.0-cp37-none-win_amd64.whl
$env:TENSORRT_DIR = "$pwd\TensorRT-8.2.3.0"
$env:path = "$env:TENSORRT_DIR\lib;" + $env:path
pip install pycuda
</code></pre>
   </td>
  </tr>
  <tr>
    <td>cudnn </td>
    <td>
    1. 从 <a href="https://developer.nvidia.com/rdp/cudnn-archive">cuDNN Archive</a> 中选择和您环境中 CPU 架构、CUDA 版本以及 TensorRT 版本配套的 cuDNN。以前文 TensorRT 安装说明为例，它需要 cudnn8.2。因此，可以下载 <a href="https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-windows-x64-v8.2.1.32.zip">CUDA 11.x cuDNN 8.2</a><br>
    2. 解压压缩包，并设置环境变量
<pre><code>
cd \the\path\of\cudnn\zip\file
Expand-Archive cudnn-11.3-windows-x64-v8.2.1.32.zip .
$env:CUDNN_DIR="$pwd\cuda"
$env:path = "$env:CUDNN_DIR\bin;" + $env:path
</code></pre>
   </td>
  </tr>
  <tr>
    <td>PPL.NN</td>
    <td>ppl.nn </td>
    <td> TODO </td>
  </tr>
  <tr>
    <td>OpenVINO</td>
    <td>openvino </td>
    <td>TODO </td>
  </tr>
  <tr>
    <td>ncnn </td>
    <td>ncnn </td>
    <td>TODO </td>
  </tr>
</tbody>
</table>

### 编译 MMDeploy

```powershell
cd \the\root\path\of\MMDeploy
$env:MMDEPLOY_DIR="$pwd"
```

#### 编译 Model Converter

如果您选择了 ONNXRuntime，TensorRT 和 ncnn 任一种推理后端，您需要编译对应的自定义算子库。

- **ONNXRuntime** 自定义算子

```powershell
mkdir build -ErrorAction SilentlyContinue
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 -DMMDEPLOY_TARGET_BACKENDS="ort" -DONNXRUNTIME_DIR="$env:ONNXRUNTIME_DIR"
cmake --build . --config Release -- /m
cmake --install . --config Release
```

- **TensorRT** 自定义算子

```powershell
mkdir build -ErrorAction SilentlyContinue
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 -DMMDEPLOY_TARGET_BACKENDS="trt" -DTENSORRT_DIR="$env:TENSORRT_DIR" -DCUDNN_DIR="$env:CUDNN_DIR"
cmake --build . --config Release -- /m
cmake --install . --config Release
```

- **ncnn** 自定义算子

  TODO

参考 [cmake 选项说明](cmake_option.md)

#### 安装 Model Converter

```powershell
cd $env:MMDEPLOY_DIR
pip install -e .
```

**注意**

- 有些依赖项是可选的。运行 `pip install -e .` 将进行最小化依赖安装。 如果需安装其他可选依赖项，请执行`pip install -r requirements/optional.txt`，
  或者 `pip install -e .[optional]`。其中，`[optional]`可以替换为：`all`、`tests`、`build` 或 `optional`。

#### 编译 SDK 和 Demos

下文展示2个构建SDK的样例，分别用 ONNXRuntime 和 TensorRT 作为推理引擎。您可以参考它们，并结合前文 SDK 的编译选项说明，激活其他的推理引擎。

- cpu + ONNXRuntime

  ```PowerShell
  cd $env:MMDEPLOY_DIR
  mkdir build -ErrorAction SilentlyContinue
  cd build
  cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
      -DMMDEPLOY_BUILD_SDK=ON `
      -DMMDEPLOY_BUILD_EXAMPLES=ON `
      -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON `
      -DMMDEPLOY_TARGET_DEVICES="cpu" `
      -DMMDEPLOY_TARGET_BACKENDS="ort" `
      -DONNXRUNTIME_DIR="$env:ONNXRUNTIME_DIR"

  cmake --build . --config Release -- /m
  cmake --install . --config Release
  ```

- cuda + TensorRT

  ```PowerShell
  cd $env:MMDEPLOY_DIR
  mkdir build
  cd build
  cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
    -DMMDEPLOY_BUILD_SDK=ON `
    -DMMDEPLOY_BUILD_EXAMPLES=ON `
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON `
    -DMMDEPLOY_TARGET_DEVICES="cuda" `
    -DMMDEPLOY_TARGET_BACKENDS="trt" `
    -Dpplcv_DIR="$env:PPLCV_DIR/pplcv-build/install/lib/cmake/ppl" `
    -DTENSORRT_DIR="$env:TENSORRT_DIR" `
    -DCUDNN_DIR="$env:CUDNN_DIR"

  cmake --build . --config Release -- /m
  cmake --install . --config Release
  ```

### 注意事项

1. Release / Debug 库不能混用。MMDeploy 要是编译 Release 版本，所有第三方依赖都要是 Release 版本。反之亦然。
