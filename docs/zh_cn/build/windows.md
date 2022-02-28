- [Windows 下构建方式](#windows-下构建方式)
  - [源码安装](#源码安装)
    - [安装构建和编译工具链](#安装构建和编译工具链)
    - [安装依赖包](#安装依赖包)
      - [安装 MMDeploy Converter 依赖](#安装-mmdeploy-converter-依赖)
      - [安装 MMDeploy SDK 依赖](#安装-mmdeploy-sdk-依赖)
      - [安装推理引擎](#安装推理引擎)
    - [编译 MMDeploy](#编译-mmdeploy)
      - [编译安装 Model Converter](#编译安装-model-converter)
        - [编译自定义算子](#编译自定义算子)
        - [安装 Model Converter](#安装-model-converter)
      - [编译 SDK](#编译-sdk)
        - [编译选项说明](#编译选项说明)
        - [编译样例](#编译样例)
        - [编译 SDK Demo](#编译-sdk-demo)
    - [注意事项](#注意事项)

---
# Windows 下构建方式

目前，MMDeploy 在 Windows 平台下仅提供源码编译安装方式。未来会提供预编译包方式。

## 源码安装
下述安装方式，均是在 **Windows 10** 下进行
### 安装构建和编译工具链
1. 下载并安装 [Visual Studio 2019](https://visualstudio.microsoft.com) 。安装时请勾选 "使用C++的桌面开发, "Windows 10 SDK <br>
2. 把 cmake 路径加入到环境变量 PATH 中, "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin" <br>
3. 如果系统中配置了 NVIDIA 显卡，根据官网教程，下载并安装 cuda toolkit。<br>
### 安装依赖包

#### 安装 MMDeploy Converter 依赖
<table>
<thead>
  <tr>
    <th>名称 </th>
    <th>安装方法 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>conda </td>
    <td>强烈建议安装conda，或者miniconda。比如， <br>https://repo.anaconda.com/miniconda/Miniconda3-py37_4.11.0-Windows-x86_64.exe <br>安装完毕后，打开系统开始菜单，输入prompt，选择并打开 anaconda powershell prompt。 <br><b>下文中的安装命令均是在 anaconda powershell 中测试验证的。</b> </td>
  </tr>
  <tr>
    <td>pytorch <br>(>=1.8.0) </td>
    <td>
    参考<a href="https://pytorch.org/get-started/locally/">pytorch官网</a>，根据系统环境, 选择合适的预编译包进行安装。比如, <br>
    <pre><code>
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    </code></pre>
    </td>
  </tr>
  <tr>
    <td>mmcv-full </td>
    <td>参考<a href="https://github.com/open-mmlab/mmcv">mmcv官网</a>，根据系统环境，选择预编译包进行安装。比如，<br>
    <pre><code>
    $env:cu_version="cu111"
    $env:torch_version="torch1.8.0"
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/$env:cu_version/$env:torch_version/index.html
    </code></pre>
    </td>
  </tr>
</tbody>
</table>


#### 安装 MMDeploy SDK 依赖
<table>
<thead>
  <tr>
    <th>名称 </th>
    <th>安装方法 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>spdlog </td>
    <td>spdlog是一个精巧的日志管理库。请参考如下命令安装： <br>
    1. 下载 https://github.com/gabime/spdlog/archive/refs/tags/v1.9.2.zip <br>
    2. 解压后，进入到文件夹 spdlog-v1.9.2 <br>
    3. 执行编译安装命令 <br>
    <pre><code>
    mkdir build
    cd build
    cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target install -j --config Release
    </code></pre>
   </td>
  </tr>
  <tr>
    <td>OpenCV </td>
    <td>
    1. 下载并安装 OpenCV 在 windows 下的预编译包: https://github.com/opencv/opencv/releases/download/4.5.5/opencv-4.5.5-vc14_vc15.exe <br>
    2. 把 OpenCV 库的路径加入到环境变量 PATH 中</td>

  </tr>
  <tr>
    <td>pplcv </td>
    <td>pplcv 是在x86和cuda平台下的高性能图像处理库。 <b>此依赖项为可选项，只有在cuda平台下，才需安装。而且，目前必须使用v0.6.2，且需要使用git clone的方式下载源码并编译安装</b><br>
    <pre><code>
    git clone --recursive git@github.com:openppl-public/ppl.cv.git
    cd ppl.cv
    git checkout tags/v0.6.2 -b v0.6.2
    ./build.bat -G "Visual Studio 16 2019" -T v142 -A x64 -DHPCC_USE_CUDA=ON -DHPCC_MSVC_MD=ON
    </code></pre>
   </td>
  </tr>
</tbody>
</table>



#### 安装推理引擎
目前，在 Windows 平台下，MMDeploy 支持 ONNXRuntime 和 TensorRT 两种推理引擎。其他推理引擎尚未进行验证，或者验证未通过。后续将陆续予以支持
<table>
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
    <td>onnxruntime </td>
    <td>
    1. 下载二进制包：https://github.com/microsoft/onnxruntime/releases/download/v1.8.0/onnxruntime-win-x64-1.8.0.zip <br>
    2. 解压到目标路径。我们使用 <code>onnxruntime_dir</code> 代表此路径 <br>
    3. 在 PATH 中增加 onnxruntime libs 路径,
    <pre><code>
    $env:path = "{onnxruntime_dir}/lib;" + $env:path
    </code></pre>
    </td>
  </tr>
  <tr>
    <td rowspan="2">TensorRT<br> </td>
    <td>TensorRT <br> </td>
    <td>
    1. 从NVIDIA官网下载二进制包, 比如，<br>
   https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.3.0/zip/TensorRT-8.2.3.0.Windows10.x86_64.cuda-11.4.cudnn8.2.zip <br>
    1. 解压二进制包到目标路径。我们使用 <code>tensorrt_dir</code> 代表此路径 <br>
    2. 安装 tensorrt 的 python package<br>
    3. 在 PATH 中增加 tensorrt libs 路径
   <pre><code>
   pip install {tensorrt_dir}/python/tensorrt-8.2.3.0-cp37-none-win_amd64.whl
   $env:path = "{tensorrt_dir}/lib;" + $env:path
   </code></pre>
   </td>
  </tr>
  <tr>
    <td>cudnn </td>
    <td>
    1. 从NVIDIA官网下载二进制包, 比如, <br>
   https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-windows-x64-v8.2.1.32.zip <br>
    1. 解压二进制包到目标路径。我们使用 <code>cudnn_dir</code> 代表此路径 <br>
    2. 在 PATH 中增加 cudnn libs 路径
   <pre><code>
   $env:path = "{cudnn_dir}/bin;" + $env:path
   </code><pre>
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

#### 编译安装 Model Converter
##### 编译自定义算子
- **ONNXRuntime** 自定义算子
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 -DMMDEPLOY_TARGET_BACKENDS="ort" -DONNXRUNTIME_DIR={onnxruntime_dir}
cmake --build . --config Release -- /maxcpucount:4
```

- **TensorRT** 自定义算子

```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 -DMMDEPLOY_TARGET_BACKENDS="trt" -DTENSORRT_DIR={tensorrt_dir} -DCUDNN_DIR={cudnn_dir}
cmake --build . --config Release -- /maxcpucount:4
```

- **ncnn** 自定义算子

  TODO

##### 安装 Model Converter
```powershell
cd root/path/of/MMDeploy
pip install -e .
```
#### 编译 SDK
##### 编译选项说明
<table>
<thead>
  <tr>
    <th>编译选项</th>
    <th>取值范围</th>
    <th>缺省值</th>
    <th>说明</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MMDEPLOY_BUILD_SDK</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>MMDeploy SDK 编译开关</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_SDK_PYTHON_API</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>MMDeploy SDK python package的编译开关</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_TEST</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>MMDeploy SDK的测试程序编译开关</td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_DEVICES</td>
    <td>{"cpu", "cuda"}</td>
    <td>cpu</td>
    <td>设置目标设备。当有多个设备时，设备名称之间使用分号隔开。 比如，-DMMDEPLOY_TARGET_DEVICES="cpu;cuda"</td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_BACKENDS</td>
    <td>{"trt", "ort", "pplnn", "ncnn", "openvino"}</td>
    <td>N/A</td>
    <td> <b>默认情况下，SDK不设置任何后端</b>, 因为它与应用场景高度相关。 当选择多个后端时， 中间使用分号隔开。比如，<pre><code>-DMMDEPLOY_TARGET_BACKENDS="trt;ort;pplnn;ncnn;openvino"</code></pre>
    构建时，几乎每个后端，都需设置一些环境变量，用来查找依赖包。<br>
    1. <b>trt</b>: 表示 TensorRT, 需要设置 TENSORRT_DIR 和 CUDNN_DIR。类似， <pre><code>-DTENSORRT_DIR={tensorrt_dir}<br>-DCUDNN_DIR={cudnn_dir}</code></pre>
    2. <b>ort</b>: 表示 ONNXRuntime，需要设置 ONNXRUNTIME_DIR。类似， <pre><code>-DONNXRUNTIME_DIR={onnxruntime_dir}</code></pre>
    3. <b>pplnn</b>: 表示 PPL.NN，需要设置 pplnn_DIR。<b>当前版本尚未验证</b> <br>
    4. <b>ncnn</b>：需要设置 ncnn_DIR。<b>当前版本尚未验证</b> <br>
    5. <b>openvino</b>: 表示 OpenVINO，需要设置 InferenceEngine_DIR。<b>当前版本尚未验证通过</b>
   </td>
  </tr>
  <tr>
    <td>MMDEPLOY_CODEBASES</td>
    <td>{"mmcls", "mmdet", "mmseg", "mmedit", "mmocr", "all"}</td>
    <td>N/A</td>
    <td>用来设置SDK后处理组件，加载OpenMMLab算法仓库的后处理功能。已支持的算法仓库有'mmcls'，'mmdet'，'mmedit'，'mmseg'和'mmocr'。如果选择多个codebase，中间使用分号隔开。比如，<code>-DMMDEPLOY_CODEBASES="mmcls;mmdet"</code>。也可以通过 <code>-DMMDEPLOY_CODEBASES=all</code> 方式，加载所有codebase。</td>
  </tr>
  <tr>
    <td>BUILD_SHARED_LIBS</td>
    <td>{ON, OFF}</td>
    <td>ON</td>
    <td>动态库的编译开关。设置OFF时，编译静态库</td>
  </tr>
</tbody>
</table>


##### 编译样例

下文展示2个构建SDK的样例，分别用于不同的运行环境。

- cpu + ONNXRuntime

  ```PowerShell
  mkdir build
  cd build
  cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
      -DMMDEPLOY_BUILD_SDK=ON `
      -DMMDEPLOY_TARGET_DEVICES="cpu" `
      -DMMDEPLOY_TARGET_BACKENDS="ort" `
      -DMMDEPLOY_CODEBASES="all" `
      -DONNXRUNTIME_DIR={onnxruntime_dir} `
      -Dspdlog_DIR={spdlog_dir}/build/install/lib/cmake/spdlog `
      -DOpenCV_DIR={opencv_dir}/build
  cmake --build . --config Release -- /maxcpucount:4
  cmake --install . --config Release
  ```

- cuda + TensorRT

  ```PowerShell
   mkdir build
   cd build
   cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
     -DMMDEPLOY_BUILD_SDK=ON `
     -DMMDEPLOY_TARGET_DEVICES="cuda" `
     -DMMDEPLOY_TARGET_BACKENDS="trt" `
     -DMMDEPLOY_CODEBASES="all" `
     -Dpplcv_DIR={pplcv_dir}/pplcv-build/install/lib/cmake/ppl `
     -DTENSORRT_DIR={tensorrt_dir} `
     -DCUDNN_DIR={cudnn_dir} `
     -Dspdlog_DIR={spdlog_dir}/build/install/lib/cmake/spdlog `
     -DOpenCV_DIR={opencv_dir}/build
   cmake --build . --config Release -- /maxcpucount:4
   cmake --install . --config Release
  ```
- 其他

  请参考上述两个示例，以及前述 SDK 的编译选项，在其他运行环境下编译 SDK

##### 编译 SDK Demo

```PowerShell
cd install/example
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
  -DMMDeploy_DIR={mmdeploy_dir}/build/install/lib/cmake/MMDeploy `
  -Dspdlog_DIR={spdlog_dir}/build/install/lib/cmake/spdlog `
  -DOpenCV_DIR={opencv_dir}/build
cmake --build . --config Release -- /maxcpucount:4

$env:path = "${mmdeploy_dir}/build/install/bin;" + $env:path

```

### 注意事项
  1. Release / Debug 库不能混用。MMDeploy要是编译Debug版本，所有第三方依赖都要是Debug版本。
