## Introduction

## Installation

### Dependencies

MMDeploy requires a compiler supporting C++17, e.g. GCC 7+, and CMake 3.14+ to build. Currently, it's tested on Linux
x86-64, more platforms will be added in the future. The following packages are required to build MMDeploy SDK

- OpenCV 3+
- spdlog 0.16+

Make sure they can be found by `find_package` in cmake. If they are not installed by your OS's package manager, you
probably need to pass their locations via `CMAKE_PREFIX_PATH` or as `*_DIR` variable.

On Ubuntu 16.04, please use the following command to install spdlog instead of `apt-get install libspdlog-dev`
```bash
wget http://archive.ubuntu.com/ubuntu/pool/universe/s/spdlog/libspdlog-dev_0.16.3-1_amd64.deb
sudo dpkg -i libspdlog-dev_0.16.3-1_amd64.deb
```

### Enabling devices

By default, only CPU device is included in the target devices. You can enable device support for other devices by
passing a semicolon separated list of device names to `MMDEPLOY_TARGET_DEVICES` variable, e.g. `"cpu;cuda"`. Currently,
the following devices are supported.

| device |  name | path setter |
|--------|-------|-------------|
|  Host  |  cpu  |    N/A      |
|  CUDA  |  cuda | CUDA_TOOLKIT_ROOT_DIR |

If you have multiple CUDA versions installed on your system, you will need to pass `CUDA_TOOLKIT_ROOT_DIR` to cmake to
specify the version.

### Enabling inference engines

**By default, no target inference engines are set**, since it's highly dependent on the use
case. `MMDEPLOY_TARGET_BACKENDS`
must be set to a semicolon separated list of inference engine names. A path to the inference engine library is also
needed. The following backends are currently supported

|   library   |  name    |   path setter   |
|-------------|----------|-----------------|
| PPL.nn      | pplnn    | pplnn_DIR       |
| ncnn        | ncnn     | ncnn_DIR        |
| ONNXRuntime | ort      | ONNXRUNTIME_DIR |
| TensorRT    | trt      | TENSORRT_DIR & CUDNN_DIR |
| OpenVINO    | openvino | InferenceEngine_DIR |

### Put it all together

The following is a recipe for building MMDeploy SDK with CPU device and ONNXRuntime support

```Bash
mkdir build && cd build
cmake \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++-7 \
    -DOpenCV_DIR=/path/to/OpenCV/lib/cmake/OpenCV \
    -Dspdlog_DIR=/path/to/spdlog/lib/cmake/spdlog \
    -DONNXRUNTIME_DIR=/path/to/onnxruntime \
    -DMMDEPLOY_TARGET_DEVICES=cpu \
    -DMMDEPLOY_TARGET_BACKENDS=ort \
    -DMMDEPLOY_CODEBASES=all
    ..
cmake --build . -- -j$(nproc) && cmake --install .
```

## Getting Started

After building & installing, the installation folder should have the following structure

```
.
└── Release
    ├── example
    │   ├── CMakeLists.txt
    │   ├── image_classification.cpp
    │   └── object_detection.cpp
    ├── include
    │   ├── c
    │   │   ├── classifier.h
    │   │   ├── common.h
    │   │   ├── detector.h
    │   │   ├── restorer.h
    │   │   ├── segmentor.h
    │   │   ├── text_detector.h
    │   │   └── text_recognizer.h
    │   └── cpp
    │       ├── archive
    │       ├── core
    │       └── experimental
    └── lib
```

where `include/c` and `include/cpp` correspond to C and C++ API respectively.

**Caution: The C++ API is highly volatile and not recommended at the moment.**

In the example directory, there are 2 examples involving classification and object detection. The examples are tested
with ONNXRuntime on CPU. More examples on more devices/backends will come once our cmake packaging code is ready.

To start with, put the corresponding ONNX model file exported for ONNXRuntime in `demo/config/resnet50_ort`
and `demo/config/retinanet_ort`. The models should be renamed as `end2end.onnx` to match the configs. The models can
be exported using [MMDeploy](https://github.com/open-mmlab/mmdeploy) or corresponding OpenMMLab codebases.
This can be done automatically when the model conversion to SDK model packaging script is ready in the future.


Here is a recipe for building & running the examples

```Bash
cd build/install/example

# path to onnxruntime ** libraries **
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib

mkdir build && cd build
cmake -DOpenCV_DIR=path/to/OpenCV/lib/cmake/OpenCV \
      -DMMDeploy_DIR=${DMMDeploy_SOURCE_ROOT_DIR}/build/install/lib/cmake/MMDeploy ..
cmake --build .

# suppress verbose logs
export SPDLOG_LEVEL=warn

# running the image classification example
./image_classification ../config/resnet50_ort ${path/to/an/image}

# running the object detection example
./object_detection ../config/retinanet_ort ${path/to/an/image}
```
