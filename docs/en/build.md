## Build MMDeploy

We provide building methods for both physical and virtual machines. For virtual machine building methods, please refer to
[how to use docker](tutorials/how_to_use_docker.md). For physical machine, please follow the steps below.

### Preparation

- Download MMDeploy

    ```bash
    git clone -b master git@github.com:open-mmlab/mmdeploy.git MMDeploy
    cd MMDeploy
    git submodule update --init --recursive
    ```

    Note:

  - If fetching submodule fails, you could get submodule manually by following instructions:

      ```bash
      git clone git@github.com:NVIDIA/cub.git third_party/cub
      cd third_party/cub
      git checkout c3cceac115

      # go back to third_party directory and git clone pybind11
      cd ..
      git clone git@github.com:pybind/pybind11.git pybind11
      cd pybind11
      git checkout 70a58c5
      ```

- Install cmake

    Install cmake>=3.14.0, you could refer to [cmake website](https://cmake.org/install) for more detailed info.

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

### Create Environment

- Create a conda virtual environment and activate it

    ```bash
    conda create -n mmdeploy python=3.7 -y
    conda activate mmdeploy
    ```

- Install PyTorch>=1.8.0, following the [official instructions](https://pytorch.org/)

    ```bash
    # CUDA 11.1
    conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    ```

- Install mmcv-full. Refer to the [guide](https://github.com/open-mmlab/mmcv#installation) for details.

    ```bash
    export cu_version=cu111 # cuda 11.1
    export torch_version=torch1.8.0
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/${cu_version}/${torch_version}/index.html
    ```



### Build backend support

Build the inference engine extension libraries you need.

- [ONNX Runtime](backends/onnxruntime.md)
- [TensorRT](backends/tensorrt.md)
- [ncnn](backends/ncnn.md)
- [pplnn](backends/pplnn.md)
- [OpenVINO](backends/openvino.md)

### Install mmdeploy

```bash
cd ${MMDEPLOY_DIR} # To mmdeploy root directory
pip install -e .
```

**Note**

- Some dependencies are optional. Simply running `pip install -e .` will only install the minimum runtime requirements.
To use optional dependencies, install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -e . [optional]`).
Valid keys for the extras field are: `all`, `tests`, `build`, `optional`.

### Build SDK

Readers can skip this chapter if you are only interested in model converter.

#### Dependencies

Currently, SDK is tested on Linux x86-64, more platforms will be added in the future. The following packages are required to build MMDeploy SDK.

Each package's installation command is given based on Ubuntu 18.04.

- OpenCV 3+

  ```bash
  sudo apt-get install libopencv-dev
  ```

- spdlog 0.16+

  ``` bash
  sudo apt-get install libspdlog-dev
  ```

  On Ubuntu 16.04, please use the following command
  ```bash
  wget http://archive.ubuntu.com/ubuntu/pool/universe/s/spdlog/libspdlog-dev_0.16.3-1_amd64.deb
  sudo dpkg -i libspdlog-dev_0.16.3-1_amd64.deb
  ```

  You can also build spdlog from its source to enjoy its latest features. But be sure to add **`-fPIC`** compilation flags at first.

- pplcv

  A high-performance image processing library of openPPL supporting x86 and cuda platforms.</br>
  It is **OPTIONAL** which only be needed if `cuda` platform is required.
  ```bash
  git clone git@github.com:openppl-public/ppl.cv.git
  cd ppl.cv
  ./build.sh cuda
  ```

- backend engines

  SDK uses the same backends as model converter does. Please follow [build backend](#build-backend-support) guide to install your interested backend.

#### Set Build Option

- Turn on SDK build switch

  `-DMMDEPLOY_BUILD_SDK=ON`


- Enabling Devices

   By default, only CPU device is included in the target devices. You can enable device support for other devices by
   passing a semicolon separated list of device names to `MMDEPLOY_TARGET_DEVICES` variable, e.g. `-DMMDEPLOY_TARGET_DEVICES="cpu;cuda"`. </br>
   Currently, the following devices are supported.

   | device |  name | path setter |
   |--------|-------|-------------|
   |  Host  |  cpu  |    N/A      |
   |  CUDA  |  cuda | CUDA_TOOLKIT_ROOT_DIR & pplcv_DIR |

   If you have multiple CUDA versions installed on your system, you will need to pass `CUDA_TOOLKIT_ROOT_DIR` to cmake to specify the version. </br>
   Meanwhile, `pplcv_DIR` has to be provided in order to build image processing operators on cuda platform.


- Enabling inference engines

   **By default, no target inference engines are set**, since it's highly dependent on the use case.
   `MMDEPLOY_TARGET_BACKENDS` must be set to a semicolon separated list of inference engine names,
   e.g. `-DMMDEPLOY_TARGET_BACKENDS="trt;ort;pplnn;ncnn;openvino"`
   A path to the inference engine library is also needed. The following backends are currently supported

   |   library   |  name    |   path setter   |
   |-------------|----------|-----------------|
   | PPL.nn      | pplnn    | pplnn_DIR       |
   | ncnn        | ncnn     | ncnn_DIR        |
   | ONNXRuntime | ort      | ONNXRUNTIME_DIR |
   | TensorRT    | trt      | TENSORRT_DIR & CUDNN_DIR |
   | OpenVINO    | openvino | InferenceEngine_DIR |

- Enabling codebase's postprocess components

  `MMDEPLOY_CODEBASES` MUST be specified by a semicolon separated list of codebase names.
  The currently supported codebases are 'mmcls', 'mmdet', 'mmedit', 'mmseg', 'mmocr'.
  Instead of listing them one by one in `MMDEPLOY_CODEBASES`, user can also pass `all` to enable all of them, i.e.,
  `-DMMDEPLOY_CODEBASES=all`


- Put it all together

  The following is a recipe for building MMDeploy SDK with cpu device and ONNXRuntime support
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

  Here is another example to build MMDeploy SDK with cuda device and TensorRT backend

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
