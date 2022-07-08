# Build for Windows

- [Build for Windows](#build-for-windows)
  - [Build From Source](#build-from-source)
    - [Install Toolchains](#install-toolchains)
    - [Install Dependencies](#install-dependencies)
      - [Install Dependencies for Model Converter](#install-dependencies-for-model-converter)
      - [Install Dependencies for SDK](#install-dependencies-for-sdk)
      - [Install Inference Engines for MMDeploy](#install-inference-engines-for-mmdeploy)
    - [Build MMDeploy](#build-mmdeploy)
      - [Build Options Spec](#build-options-spec)
      - [Build Model Converter](#build-model-converter)
        - [Build Custom Ops](#build-custom-ops)
        - [Install Model Converter](#install-model-converter)
      - [Build SDK](#build-sdk)
      - [Build Demo](#build-demo)
    - [Note](#note)

______________________________________________________________________

Currently, MMDeploy only provides build-from-source method for windows platform. Prebuilt package will be released in the future.

## Build From Source

All the commands listed in the following chapters are verified on **Windows 10**.

### Install Toolchains

1. Download and install [Visual Studio 2019](https://visualstudio.microsoft.com)
2. Add the path of `cmake` to the environment variable `PATH`, i.e., "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin"
3. Install cuda toolkit if NVIDIA gpu is available. You can refer to the official [guide](https://developer.nvidia.com/cuda-downloads).

### Install Dependencies

#### Install Dependencies for Model Converter

<table class="docutils">
<thead>
  <tr>
    <th>NAME </th>
    <th>INSTALLATION </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>conda </td>
    <td> Please install conda according to the official <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">guide</a>. <br>
 After installation, open <code>anaconda powershell prompt</code> under the Start Menu <b>as the administrator</b>, because: <br>
1. <b>All the commands listed in the following text are verified in anaconda powershell </b><br>
2. <b>As an administrator, you can install the thirdparty libraries to the system path so as to simplify MMDeploy build command</b><br>
Note: if you are familiar with how cmake works, you can also use <code>anaconda powershell prompt</code> as an ordinary user.
    </td>
  </tr>
  <tr>
    <td>PyTorch <br>(>=1.8.0) </td>
    <td>
    Install PyTorch>=1.8.0 by following the <a href="https://pytorch.org/">official instructions</a>. Be sure the CUDA version PyTorch requires matches that in your host.
<pre><code>
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
</code></pre>
    </td>
  </tr>
  <tr>
    <td>mmcv-full </td>
    <td>Install mmcv-full as follows. Refer to the <a href="https://github.com/open-mmlab/mmcv#installation">guide</a> for details.
<pre><code>
$env:cu_version="cu111"
$env:torch_version="torch1.8.0"
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/${cu_version}/${torch_version}/index.html
</code></pre>
    </td>
  </tr>
</tbody>
</table>

#### Install Dependencies for SDK

You can skip this chapter if you are only interested in the model converter.

<table class="docutils">
<thead>
  <tr>
    <th>NAME </th>
    <th>INSTALLATION </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>OpenCV<br>(>=3.0) </td>
    <td>
    1. Find and download OpenCV 3+ for windows from <a href="https://github.com/opencv/opencv/releases">here</a>.<br>
    2. You can download the prebuilt package and install it to the target directory. Or you can build OpenCV from its source. <br>
    3. Find where <code>OpenCVConfig.cmake</code> locates in the installation directory. And export its path to the environment variable <code>PATH</code> like this,
<pre><code>$env:path = "\the\path\where\OpenCVConfig.cmake\locates;" + "$env:path"</code></pre>
    </td>
  </tr>
  <tr>
    <td>pplcv </td>
    <td>A high-performance image processing library of openPPL.<br>
  <b>It is optional which only be needed if <code>cuda</code> platform is required.
  Now, MMDeploy supports v0.6.2 and has to use <code>git clone</code> to download it.</b><br>
<pre><code>
git clone https://github.com/openppl-public/ppl.cv.git
cd ppl.cv
git checkout tags/v0.6.2 -b v0.6.2
$env:PPLCV_DIR = "$pwd"
mkdir pplcv-build
cd pplcv-build
cmake .. -G "Visual Studio 16 2019" -T v142 -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DHPCC_USE_CUDA=ON -DHPCC_MSVC_MD=ON
cmake --build . --config Release -- /m
cmake --install . --config Release
cd ../..
</code></pre>
   </td>
  </tr>
</tbody>
</table>

#### Install Inference Engines for MMDeploy

Both MMDeploy's model converter and SDK share the same inference engines.
You can select your interested inference engines and do the installation by following the given commands.

**Currently, MMDeploy only verified ONNXRuntime and TensorRT for windows platform**.
As for the rest, MMDeploy will support them in the future.

<table class="docutils">
<thead>
  <tr>
    <th>NAME</th>
    <th>PACKAGE</th>
    <th>INSTALLATION </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ONNXRuntime</td>
    <td>onnxruntime<br>(>=1.8.1) </td>
    <td>
    1. Install python package
<pre><code>pip install onnxruntime==1.8.1</code></pre>
    2. Download the windows prebuilt binary package from <a href="https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1">here</a>. Extract it and export environment variables as below:
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
   1. Login <a href="https://www.nvidia.com/">NVIDIA</a> and download the TensorRT tar file that matches the CPU architecture and CUDA version you are using from <a href="https://developer.nvidia.com/nvidia-tensorrt-download">here</a>. Follow the <a href="https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar">guide</a> to install TensorRT. <br>
   2. Here is an example of installing TensorRT 8.2 GA Update 2 for Windows x86_64 and CUDA 11.x that you can refer to. <br> First of all, click <a href="https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.3.0/zip/TensorRT-8.2.3.0.Windows10.x86_64.cuda-11.4.cudnn8.2.zip">here</a> to download CUDA 11.x TensorRT 8.2.3.0 and then install it and other dependency like below:
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
    <td>cuDNN </td>
    <td>
    1. Download cuDNN that matches the CPU architecture, CUDA version and TensorRT version you are using from <a href="https://developer.nvidia.com/rdp/cudnn-archive"> cuDNN Archive</a>. <br>
In the above TensorRT's installation example, it requires cudnn8.2. Thus, you can download <a href="https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-windows-x64-v8.2.1.32.zip">CUDA 11.x cuDNN 8.2</a><br>
    2. Extract the zip file and set the environment variables
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
    <td>TODO </td>
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

### Build MMDeploy

```powershell
cd \the\root\path\of\MMDeploy
$env:MMDEPLOY_DIR="$pwd"
```

#### Build Options Spec

<table class="docutils">
<thead>
  <tr>
    <th>NAME</th>
    <th>VALUE</th>
    <th>DEFAULT</th>
    <th>REMARK</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MMDEPLOY_BUILD_SDK</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Switch to build MMDeploy SDK</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_SDK_PYTHON_API</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>switch to build MMDeploy SDK python package</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_TEST</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Switch to build MMDeploy SDK unittest cases</td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_DEVICES</td>
    <td>{"cpu", "cuda"}</td>
    <td>cpu</td>
    <td>Enable target device. You can enable more by
   passing a semicolon separated list of device names to <code>MMDEPLOY_TARGET_DEVICES</code> variable, e.g. <code>-DMMDEPLOY_TARGET_DEVICES="cpu;cuda"</code> </td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_BACKENDS</td>
    <td>{"trt", "ort", "pplnn", "ncnn", "openvino"}</td>
    <td>N/A</td>
    <td>Enabling inference engine. <b>By default, no target inference engine is set, since it highly depends on the use case.</b> When more than one engine are specified, it has to be set with a semicolon separated list of inference backend names, e.g. <pre><code>-DMMDEPLOY_TARGET_BACKENDS="trt;ort;pplnn;ncnn;openvino"</code></pre>
    After specifying the inference engine, it's package path has to be passed to cmake as follows, <br>
    1. <b>trt</b>: TensorRT. <code>TENSORRT_DIR</code> and <code>CUDNN_DIR</code> are needed.
<pre><code>
-DTENSORRT_DIR=$env:TENSORRT_DIR
-DCUDNN_DIR=$env:CUDNN_DIR
</code></pre>
    2. <b>ort</b>: ONNXRuntime. <code>ONNXRUNTIME_DIR</code> is needed.
<pre><code>-DONNXRUNTIME_DIR=$env:ONNXRUNTIME_DIR</code></pre>
    3. <b>pplnn</b>: PPL.NN. <code>pplnn_DIR</code> is needed. MMDeploy hasn't verified it yet.
    4. <b>ncnn</b>: ncnn. <code>ncnn_DIR</code> is needed. MMDeploy hasn't verified it yet.
    5. <b>openvino</b>: OpenVINO. <code>InferenceEngine_DIR</code> is needed. MMDeploy hasn't verified it yet.
   </td>
  </tr>
  <tr>
    <td>MMDEPLOY_CODEBASES</td>
    <td>{"mmcls", "mmdet", "mmseg", "mmedit", "mmocr", "all"}</td>
    <td>all</td>
    <td>Enable codebase's postprocess modules. You can provide a semicolon separated list of codebase names to enable them. Or you can pass <code>all</code> to enable them all, i.e., <code>-DMMDEPLOY_CODEBASES=all</code></td>
  </tr>
  <tr>
    <td>MMDEPLOY_SHARED_LIBS</td>
    <td>{ON, OFF}</td>
    <td>ON</td>
    <td>Switch to build shared library or static library of MMDeploy SDK</td>
  </tr>
</tbody>
</table>

#### Build Model Converter

##### Build Custom Ops

If one of inference engines among ONNXRuntime, TensorRT and ncnn is selected, you have to build the corresponding custom ops.

- **ONNXRuntime** Custom Ops

```powershell
mkdir build -ErrorAction SilentlyContinue
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 -DMMDEPLOY_TARGET_BACKENDS="ort" -DONNXRUNTIME_DIR="$env:ONNXRUNTIME_DIR"
cmake --build . --config Release -- /m
cmake --install . --config Release
```

- **TensorRT** Custom Ops

```powershell
mkdir build -ErrorAction SilentlyContinue
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 -DMMDEPLOY_TARGET_BACKENDS="trt" -DTENSORRT_DIR="$env:TENSORRT_DIR" -DCUDNN_DIR="$env:CUDNN_DIR"
cmake --build . --config Release -- /m
cmake --install . --config Release
```

- **ncnn** Custom Ops

  TODO

##### Install Model Converter

```powershell
cd $env:MMDEPLOY_DIR
pip install -e .
```

**Note**

- Some dependencies are optional. Simply running `pip install -e .` will only install the minimum runtime requirements.
  To use optional dependencies, install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -e .[optional]`).
  Valid keys for the extras field are: `all`, `tests`, `build`, `optional`.

#### Build SDK

MMDeploy provides two recipes as shown below for building SDK with ONNXRuntime and TensorRT as inference engines respectively.
You can also activate other engines after the model.

- cpu + ONNXRuntime

  ```PowerShell
  cd $env:MMDEPLOY_DIR
  mkdir build -ErrorAction SilentlyContinue
  cd build
  cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
      -DMMDEPLOY_BUILD_SDK=ON `
      -DMMDEPLOY_TARGET_DEVICES="cpu" `
      -DMMDEPLOY_TARGET_BACKENDS="ort" `
      -DMMDEPLOY_CODEBASES="all" `
      -DONNXRUNTIME_DIR="$env:ONNXRUNTIME_DIR"

  cmake --build . --config Release -- /m
  cmake --install . --config Release
  ```

- cuda + TensorRT

  ```PowerShell
  cd $env:MMDEPLOY_DIR
  mkdir build -ErrorAction SilentlyContinue
  cd build
  cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
    -DMMDEPLOY_BUILD_SDK=ON `
    -DMMDEPLOY_TARGET_DEVICES="cuda" `
    -DMMDEPLOY_TARGET_BACKENDS="trt" `
    -DMMDEPLOY_CODEBASES="all" `
    -Dpplcv_DIR="$env:PPLCV_DIR/pplcv-build/install/lib/cmake/ppl" `
    -DTENSORRT_DIR="$env:TENSORRT_DIR" `
    -DCUDNN_DIR="$env:CUDNN_DIR"

  cmake --build . --config Release -- /m
  cmake --install . --config Release
  ```

#### Build Demo

```PowerShell
cd $env:MMDEPLOY_DIR\build\install\example
mkdir build -ErrorAction SilentlyContinue
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
  -DMMDeploy_DIR="$env:MMDEPLOY_DIR/build/install/lib/cmake/MMDeploy"

cmake --build . --config Release -- /m

$env:path = "$env:MMDEPLOY_DIR/build/install/bin;" + $env:path
```

### Note

1. Release / Debug libraries can not be mixed. If MMDeploy is built with Release mode, all its dependent thirdparty libraries have to be built in Release mode too and vice versa.
