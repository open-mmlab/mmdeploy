# TorchScript 支持情况

## Introduction of TorchScript

**TorchScript** a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency. Check the [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) for more details.

## Build custom ops

### Prerequisite

- Download libtorch from the official website [here](https://pytorch.org/get-started/locally/).

*Please note that only **Pre-cxx11 ABI** and **version 1.8.1+** on Linux platform are supported by now.*

For previous versions of libtorch, users can find through the [issue comment](https://github.com/pytorch/pytorch/issues/40961#issuecomment-1017317786). Libtorch1.8.1+cu111 as an example, extract it, expose `Torch_DIR` and add the lib path to `LD_LIBRARY_PATH` as below:

```bash
wget https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.8.1%2Bcu111.zip

unzip libtorch-shared-with-deps-1.8.1+cu111.zip
cd libtorch
export Torch_DIR=$(pwd)
export LD_LIBRARY_PATH=$Torch_DIR/lib:$LD_LIBRARY_PATH
```

Note:

- If you want to save libtorch env variables to bashrc, you could run

  ```bash
  echo '# set env for libtorch' >> ~/.bashrc
  echo "export Torch_DIR=${Torch_DIR}" >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=$Torch_DIR/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
  source ~/.bashrc
  ```

### Build on Linux

```bash
cd ${MMDEPLOY_DIR} # To MMDeploy root directory
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=torchscript -DTorch_DIR=${Torch_DIR} ..
make -j$(nproc)
```

## How to convert a model

- You could follow the instructions of tutorial [How to convert model](../02-how-to-run/convert_model.md)

## SDK backend

TorchScript SDK backend may be built by passing `-DMMDEPLOY_TORCHSCRIPT_SDK_BACKEND=ON` to `cmake`.

Notice that `libtorch` is sensitive to C++ ABI versions. On platforms defaulted to C++11 ABI (e.g. Ubuntu 16+) one may
pass `-DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"` to `cmake` to use pre-C++11 ABI for building. In this case all
dependencies with ABI sensitive interfaces (e.g. OpenCV) must be built with pre-C++11 ABI.

## FAQs

- Error: `projects/thirdparty/libtorch/share/cmake/Caffe2/Caffe2Config.cmake:96 (message):Your installed Caffe2 version uses cuDNN but I cannot find the cuDNN libraries.  Please set the proper cuDNN prefixes and / or install cuDNN.`

  May export CUDNN_ROOT=/root/path/to/cudnn to resolve the build error.
