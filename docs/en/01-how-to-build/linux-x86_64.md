# Build for Linux-x86_64

- [Build for Linux-x86_64](#build-for-linux-x86_64)
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

______________________________________________________________________

## Install Toolchains

- cmake

  **Make sure cmake version >= 3.14.0**. The below script shows how to install cmake 3.20.0. You can find more versions [here](https://cmake.org/install).

  ```bash
  wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.tar.gz
  tar -xzvf cmake-3.20.0-linux-x86_64.tar.gz
  sudo ln -sf $(pwd)/cmake-3.20.0-linux-x86_64/bin/* /usr/bin/
  ```

- GCC 7+

  MMDeploy requires compilers that support C++17.

  ```bash
  # Add repository if ubuntu < 18.04
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test
  sudo apt-get update
  sudo apt-get install gcc-7
  sudo apt-get install g++-7
  ```

## Install Dependencies

### Install Dependencies for Model Converter

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
    <td>Please install conda according to the official <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">guide</a>. <br>
    Create a conda virtual environment and activate it. <br>
<pre><code>
conda create -n mmdeploy python=3.7 -y
conda activate mmdeploy
</code></pre>
    </td>
  </tr>
  <tr>
    <td>PyTorch <br>(>=1.8.0) </td>
    <td>
    Install PyTorch>=1.8.0 by following the <a href="https://pytorch.org/">official instructions</a>. Be sure the CUDA version PyTorch requires matches that in your host.
<pre><code>
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
</code></pre>
    </td>
  </tr>
  <tr>
    <td>mmcv-full </td>
    <td>Install mmcv-full as follows. Refer to the <a href="https://github.com/open-mmlab/mmcv#installation">guide</a> for details.
<pre><code>
export cu_version=cu111 # cuda 11.1
export torch_version=torch1.8
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/${cu_version}/${torch_version}/index.html
</code></pre>
    </td>
  </tr>
</tbody>
</table>

### Install Dependencies for SDK

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
    On Ubuntu >=18.04,
<pre><code>
sudo apt-get install libopencv-dev
</code></pre>
    On Ubuntu 16.04, OpenCV has to be built from the source code. Please refer to the <a href="https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html">guide</a>.
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
export PPLCV_DIR=$(pwd)
git checkout tags/v0.6.2 -b v0.6.2
./build.sh cuda
</code></pre>
   </td>
  </tr>
</tbody>
</table>

### Install Inference Engines for MMDeploy

Both MMDeploy's model converter and SDK share the same inference engines.

You can select you interested inference engines and do the installation by following the given commands.

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
    2. Download the linux prebuilt binary package from <a href="https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1">here</a>.  Extract it and export environment variables as below:
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
   1. Login <a href="https://www.nvidia.com/">NVIDIA</a> and download the TensorRT tar file that matches the CPU architecture and CUDA version you are using from <a href="https://developer.nvidia.com/nvidia-tensorrt-download">here</a>. Follow the <a href="https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar">guide</a> to install TensorRT. <br>
   2. Here is an example of installing TensorRT 8.2 GA Update 2 for Linux x86_64 and CUDA 11.x that you can refer to. First of all, click <a href="https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.3.0/tars/tensorrt-8.2.3.0.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz">here</a> to download CUDA 11.x TensorRT 8.2.3.0 and then install it and other dependency like below:
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
    1. Download cuDNN that matches the CPU architecture, CUDA version and TensorRT version you are using from <a href="https://developer.nvidia.com/rdp/cudnn-archive"> cuDNN Archive</a>. <br>
In the above TensorRT's installation example, it requires cudnn8.2. Thus, you can download <a href="https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-linux-x64-v8.2.1.32.tgz">CUDA 11.x cuDNN 8.2</a><br>
    2. Extract the compressed file and set the environment variables
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
    1. Please follow the <a href="https://github.com/openppl-public/ppl.nn/blob/master/docs/en/building-from-source.md">guide</a> to build <code>ppl.nn</code> and install <code>pyppl</code>.<br>
    2. Export pplnn's root path to environment variable
<pre><code>
cd ppl.nn
export PPLNN_DIR=$(pwd)
</code></pre>
    </td>
  </tr>
  <tr>
    <td>OpenVINO</td>
    <td>openvino </td>
    <td>1. Install <a href="https://docs.openvino.ai/2021.4/get_started.html">OpenVINO</a> package
<pre><code>
pip install openvino-dev
</code></pre>
2. <b>Optional</b>. If you want to use OpenVINO in MMDeploy SDK, please install and configure it by following the <a href="https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_linux.html#install-openvino">guide</a>.
    </td>
  </tr>
  <tr>
    <td>ncnn </td>
    <td>ncnn </td>
    <td>1. Download and build ncnn according to its <a href="https://github.com/Tencent/ncnn/wiki/how-to-build">wiki</a>.
Make sure to enable <code>-DNCNN_PYTHON=ON</code> in your build command. <br>
2. Export ncnn's root path to environment variable
<pre><code>
cd ncnn
export NCNN_DIR=$(pwd)
export LD_LIBRARY_PATH=${NCNN_DIR}/build/install/lib/:$LD_LIBRARY_PATH
</code></pre>
3. Install pyncnn
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
</tbody>
</table>

Note: <br>
If you want to make the above environment variables permanent, you could add them to <code>~/.bashrc</code>. Take the ONNXRuntime for example,

```bash
echo '# set env for onnxruntime' >> ~/.bashrc
echo "export ONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

## Build MMDeploy

```bash
cd /the/root/path/of/MMDeploy
export MMDEPLOY_DIR=$(pwd)
```

### Build Options Spec

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
    <td>{"trt", "ort", "pplnn", "ncnn", "openvino", "torchscript"}</td>
    <td>N/A</td>
    <td>Enabling inference engine. <b>By default, no target inference engine is set, since it highly depends on the use case.</b> When more than one engine are specified, it has to be set with a semicolon separated list of inference backend names, e.g. <pre><code>-DMMDEPLOY_TARGET_BACKENDS="trt;ort;pplnn;ncnn;openvino"</code></pre>
    After specifying the inference engine, it's package path has to be passed to cmake as follows, <br>
    1. <b>trt</b>: TensorRT. <code>TENSORRT_DIR</code> and <code>CUDNN_DIR</code> are needed.
<pre><code>
-DTENSORRT_DIR=${TENSORRT_DIR}
-DCUDNN_DIR=${CUDNN_DIR}
</code></pre>
    2. <b>ort</b>: ONNXRuntime. <code>ONNXRUNTIME_DIR</code> is needed.
<pre><code>-DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}</code></pre>
    3. <b>pplnn</b>: PPL.NN. <code>pplnn_DIR</code> is needed.
<pre><code>-Dpplnn_DIR=${PPLNN_DIR}</code></pre>
    4. <b>ncnn</b>: ncnn. <code>ncnn_DIR</code> is needed.
<pre><code>-Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn</code></pre>
    5. <b>openvino</b>: OpenVINO. <code>InferenceEngine_DIR</code> is needed.
<pre><code>-DInferenceEngine_DIR=${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/share</code></pre>
    6. <b>torchscript</b>: TorchScript. <code>Torch_DIR</code> is needed.
<pre><code>-DTorch_DIR=${Torch_DIR}</code></pre>
Currently, <b>The Model Converter supports torchscript, but SDK doesn't</b>.
   </td>
  </tr>
  <tr>
    <td>MMDEPLOY_CODEBASES</td>
    <td>{"mmcls", "mmdet", "mmseg", "mmedit", "mmocr", "all"}</td>
    <td>all</td>
    <td>Enable codebase's postprocess modules. You can provide a semicolon separated list of codebase names to enable them, e.g., <code>-DMMDEPLOY_CODEBASES="mmcls;mmdet"</code>. Or you can pass <code>all</code> to enable them all, i.e., <code>-DMMDEPLOY_CODEBASES=all</code></td>
  </tr>
  <tr>
    <td>MMDEPLOY_SHARED_LIBS</td>
    <td>{ON, OFF}</td>
    <td>ON</td>
    <td>Switch to build shared library or static library of MMDeploy SDK</td>
  </tr>
</tbody>
</table>

### Build Model Converter

#### Build Custom Ops

If one of inference engines among ONNXRuntime, TensorRT, ncnn and libtorch is selected, you have to build the corresponding custom ops.

- **ONNXRuntime** Custom Ops

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
  make -j$(nproc) && make install
  ```

- **TensorRT** Custom Ops

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} -DCUDNN_DIR=${CUDNN_DIR} ..
  make -j$(nproc) && make install
  ```

- **ncnn** Custom Ops

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=ncnn -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn ..
  make -j$(nproc) && make install
  ```

- **TorchScript** Custom Ops

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=torchscript -DTorch_DIR=${Torch_DIR} ..
  make -j$(nproc) && make install
  ```

#### Install Model Converter

```bash
cd ${MMDEPLOY_DIR}
pip install -e .
```

**Note**

- Some dependencies are optional. Simply running `pip install -e .` will only install the minimum runtime requirements.
  To use optional dependencies, install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -e .[optional]`).
  Valid keys for the extras field are: `all`, `tests`, `build`, `optional`.

### Build SDK

MMDeploy provides two recipes as shown below for building SDK with ONNXRuntime and TensorRT as inference engines respectively.
You can also activate other engines after the model.

- cpu + ONNXRuntime

  ```Bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake .. \
      -DCMAKE_CXX_COMPILER=g++-7 \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
      -DMMDEPLOY_TARGET_DEVICES=cpu \
      -DMMDEPLOY_TARGET_BACKENDS=ort \
      -DMMDEPLOY_CODEBASES=all \
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
      -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
      -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
      -DMMDEPLOY_TARGET_BACKENDS=trt \
      -DMMDEPLOY_CODEBASES=all \
      -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl \
      -DTENSORRT_DIR=${TENSORRT_DIR} \
      -DCUDNN_DIR=${CUDNN_DIR}

  make -j$(nproc) && make install
  ```

### Build Demo

```Bash
cd ${MMDEPLOY_DIR}/build/install/example
mkdir -p build && cd build
cmake .. -DMMDeploy_DIR=${MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy
make -j$(nproc)
```
