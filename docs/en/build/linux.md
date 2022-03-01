# Build for Linux-x86_64

- [Build for Linux-x86_64](#build-for-linux-x86_64)
  - [Dockerfile (RECOMMENDED)](#dockerfile-recommended)
  - [Build From Source](#build-from-source)
    - [Install Toolchains](#install-toolchains)
    - [Install Dependencies](#install-dependencies)
      - [Install Dependencies for Model Converter](#install-dependencies-for-model-converter)
      - [Install Dependencies for SDK](#install-dependencies-for-sdk)
      - [Install Inference Engines for MMDeploy](#install-inference-engines-for-mmdeploy)
    - [Build MMDeploy](#build-mmdeploy)
      - [Build Model Converter](#build-model-converter)
        - [Build Custom Ops](#build-custom-ops)
        - [Install Model Converter](#install-model-converter)
      - [Build SDK](#build-sdk)
        - [Build Options](#build-options)
        - [Build SDK Libraries](#build-sdk-libraries)
        - [Build SDK Demo](#build-sdk-demo)

--- 
MMDeploy provides two build ways for linux-x86_64 platform, including dockerfile and build from source.

## Dockerfile (RECOMMENDED) 
please refer to
[how to use docker](tutorials/how_to_use_docker.md).

## Build From Source

### Install Toolchains
  
- cmake

    **Make sure cmake version >= 3.14.0**. If not, you can follow instructions below to install cmake 3.20.0. For more versions of cmake, please refer to [cmake website](https://cmake.org/install).

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

### Install Dependencies
#### Install Dependencies for Model Converter

<table>
<thead>
  <tr>
    <th>NAME </th>
    <th>INSTALLATION </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>conda </td>
    <td>Please install conda by following its official website <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">here</a>. <br> Create a conda virtual environment and activate it. <br>
    <pre><code>
    conda create -n mmdeploy python=3.7 -y
    conda activate mmdeploy
    </code></pre>
    </td>
  </tr>
  <tr>
    <td>pytorch <br>(>=1.8.0) </td>
    <td>
    Choose an appropriate PyTorch package from <a href="https://pytorch.org/get-started/locally/">here</a>. Make sure that your compilation CUDA version and runtime CUDA version match. e.g., If you have CUDA 11.1 installed under <code>/usr/local/cuda</code>, you can install pytorch 1.8 like, <br>
    <pre><code>
    conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    </code></pre>
    </td>
  </tr>
  <tr>
    <td>mmcv-full </td>
    <td>Currently, MMDeploy supports mmcv-full <b>v1.4.0</b>. As shown in below, you can replace <code>{cu_version}</code>  and <code>{torch_version}</code> in the url to your desire one. See <a href="https://github.com/open-mmlab/mmcv#installation">here</a> for different versions of MMCV compatible to different PyTorch and CUDA versions.<br>
    <pre><code>
    export cu_version=cu111 # cuda 11.1
    export torch_version=torch1.8
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/${cu_version}/${torch_version}/index.html
    </code></pre>
    </td>
  </tr>
</tbody>
</table>

#### Install Dependencies for SDK

You can skip this chapter if only interested in model converter.
<table>
<thead>
  <tr>
    <th>NAME </th>
    <th>INSTALLATION </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>spdlog </td>
    <td>
    On Ubuntu >=18.04,
    <pre><code>
    sudo apt-get install libspdlog-dev
    </code></pre>
    On Ubuntu 16.04,
    <pre><code>
    wget http://archive.ubuntu.com/ubuntu/pool/universe/s/spdlog/libspdlog-dev_0.16.3-1_amd64.deb
    sudo dpkg -i libspdlog-dev_0.16.3-1_amd64.deb
    </code></pre> <br>
    You can also build spdlog from its source to enjoy its latest features. But be sure to open cmake option <code>-DCMAKE_POSITION_INDEPENDENT_CODE=ON</code>.
   </td>
  </tr>
  <tr>
    <td>OpenCV<br>(>=3.0) </td>
    <td>
    <pre><code>
    sudo apt-get install libopencv-dev
    </code></pre>
    </td>

  </tr>
  <tr>
    <td>pplcv </td>
    <td>A high-performance image processing library of openPPL supporting x86 and cuda platforms.</br>
  <b>It is optional which only be needed if <code>cuda</code> platform is required.
  Now, MMDeploy supports v0.6.2 and has to use <code>git clone</code> to download it.</b><br>
    <pre><code>
    git clone https://github.com/openppl-public/ppl.cv.git
    cd ppl.cv
    git checkout tags/v0.6.2 -b v0.6.2
    ./build.sh cuda
    </code></pre>
   </td>
  </tr>
</tbody>
</table>

#### Install Inference Engines for MMDeploy

Both MMDeploy's model converter and SDK share the same inference engines.

Users can select their interested inference engines and do the installation by following the command.

   
### Build MMDeploy
#### Build Model Converter

##### Build Custom Ops
If one of inference backend, such as ONNXRuntime, TensorRT and ncnn is selected, you have to build the corresponding custom ops.

- Build ONNXRuntime Custom Ops
   
  ```bash
  cd ${MMDEPLOY_DIR} # To MMDeploy root directory
  mkdir -p build && cd build
  cmake -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
  make -j$(nproc)
  ```

- Build TensorRT Custom Ops
   
  ```bash
  cd ${MMDEPLOY_DIR} # To MMDeploy root directory
  mkdir -p build && cd build
  cmake -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} ..
  make -j$(nproc)
  ```

- Build ncnn Custom Ops
   
  ```bash
  cd ${MMDEPLOY_DIR} # To MMDeploy root directory
  mkdir -p build && cd build
  cmake -DMMDEPLOY_TARGET_BACKENDS=ncnn -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn ..
  make -j$(nproc)
  ```

##### Install Model Converter

```bash
cd ${MMDEPLOY_DIR}
pip install -e .
```
#### Build SDK

##### Build Options
<table>
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
    <td>Enabling inference backends. <b>By default, no target inference backend is set, since it highly depends on the use case.</b> When more than one backend are specified, it has to be set with a semicolon separated list of inference backend names, e.g. <pre><code>-DMMDEPLOY_TARGET_BACKENDS="trt;ort;pplnn;ncnn;openvino"</code></pre>
    After specifying backend, it's package path has to be passed to cmake as follows, <br>
    1. <b>trt</b>: TensorRT backend <pre><code>-DTENSORRT_DIR={tensorrt_dir}<br>-DCUDNN_DIR={cudnn_dir}</code></pre>
    2. <b>ort</b>: ONNXRuntime backend <pre><code>-DONNXRUNTIME_DIR={onnxruntime_dir}</code></pre>
    3. <b>pplnn</b>: PPL.NN backend <pre><code>-Dpplnn_DIR={pplnn_dir}</code></pre><br>
    4. <b>ncnn</b>: ncnn backend <pre><code> -Dncnn_DIR={ncnn_dir}</code></pre> <br>
    5. <b>openvino</b>: OpenVINO backend <pre><code> -DInferenceEngine_DIR={openvino_dir}</b>
   </td>
  </tr>
  <tr>
    <td>MMDEPLOY_CODEBASES</td>
    <td>{"mmcls", "mmdet", "mmseg", "mmedit", "mmocr", "all"}</td>
    <td>N/A</td>
    <td>Enable codebase's postprocess modules. It MUST be set by a semicolon separated list of codebase names. The currently supported codebases are 'mmcls', 'mmdet', 'mmedit', 'mmseg', 'mmocr'. Instead of listing them one by one, you can also pass <code>all</code> to enable them all, i.e., <code>-DMMDEPLOY_CODEBASES=all</code></td>
  </tr>
  <tr>
    <td>BUILD_SHARED_LIBS</td>
    <td>{ON, OFF}</td>
    <td>ON</td>
    <td>Switch to build shared library or static library of MMDeploy SDK</td>
  </tr>
</tbody>
</table>

##### Build SDK Libraries
MMDeploy provides two recipes as shown below for building SDK with ONNXRuntime and TensorRT as inference backends respectively. You can follow them to invoke other inference backends.

- cpu + ONNXRuntime
  ```Bash
  mkdir build && cd build
  cmake .. \
      -DCMAKE_CXX_COMPILER=g++-7 \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
      -DMMDEPLOY_TARGET_DEVICES=cpu \
      -DMMDEPLOY_TARGET_BACKENDS=ort \
      -DMMDEPLOY_CODEBASES=all \
      -DONNXRUNTIME_DIR={onnxruntime_dir}

  cmake --build . -- -j$(nproc) && cmake --install .
  ```

- cuda + TensorRT
  ```Bash
   mkdir build && cd build
   cmake .. \
     -DCMAKE_CXX_COMPILER=g++-7 \
     -DMMDEPLOY_BUILD_SDK=ON \
     -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
     -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
     -DMMDEPLOY_TARGET_BACKENDS=trt \
     -DMMDEPLOY_CODEBASES=all \
     -Dpplcv_DIR={pplcv_dir}/cuda-build/install/lib/cmake/ppl \
     -DTENSORRT_DIR={tensorrt_dir} \
     -DCUDNN_DIR={cudnn_dir}

   cmake --build . -- -j$(nproc) && cmake --install .
  ```

##### Build SDK Demo

```Bash
cd build/install/example
```
