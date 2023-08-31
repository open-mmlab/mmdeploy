# Build for macOS-arm64

- [Build for macOS-arm64](#build-for-macos-arm64)
  - [Install Toolchains](#install-toolchains)
  - [Install Dependencies](#install-dependencies)
    - [Install Dependencies for Model Converter](#install-dependencies-for-model-converter)
    - [Install Dependencies for SDK](#install-dependencies-for-sdk)
    - [Install Inference Engines for MMDeploy](#install-inference-engines-for-mmdeploy)
  - [Build MMDeploy](#build-mmdeploy)
    - [Build Model Converter](#build-model-converter)
    - [Install Model Converter](#install-model-converter)
    - [Build SDK and Demo](#build-sdk-and-demo)

## Install Toolchains

- cmake

  ```
  brew install cmake
  ```

- clang

  install Xcode or Command Line Tools

  ```
  xcode-select --install
  ```

## Install Dependencies

### Install Dependencies for Model Converter

Please refer to [get_started](../get_started.md) to install conda.

```bash
# install pytorch & mmcv
conda install pytorch==1.9.0 torchvision==0.10.0 -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc2"
```

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
<pre><code>
brew install opencv
</code></pre>
    </td>
</tbody>
</table>

### Install Inference Engines for MMDeploy

Both MMDeploy's model converter and SDK share the same inference engines.

You can select you interested inference engines and do the installation by following the given commands.

This document focus on Core ML. The installation of ONNX Runtime, ncnn and TorchScript is similar to the linux platform, please refer to the document [linux-x86_64](linux-x86_64.md) for installation.

The TorchScript model is used as the IR in the conversion process of the Core ML model. In order to support the custom operator in some models like detection models in mmdet, libtorch needs to be installed.

<table  class="docutils">
<thead>
  <tr>
    <th>NAME</th>
    <th>PACKAGE</th>
    <th>INSTALLATION</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Core ML</td>
    <td>coremltools</td>
    <td>
<pre><code>
pip install coremltools==6.3
</code></pre>
    </td>
  </tr>
  <tr>
  <td>TorchScript</td>
  <td>libtorch</td>
  <td>
  1. Libtorch doesn't provide prebuilt arm library for macOS, so you need to compile it yourself. Please note that the version of libtorch must be consistent with the version of pytorch. <br>
  2. Take LibTorch 1.9.0 as an example. You can install it like this:
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

## Build MMDeploy

```bash
cd /the/root/path/of/MMDeploy
export MMDEPLOY_DIR=$(pwd)
```

### Build Model Converter

- **Core ML**

  Core ML uses torchscript as IR, to convert models in some codebases like mmdet, you need to compile torchscript custom operators

- **torchscript** custom operators

  ```bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake -DMMDEPLOY_TARGET_BACKENDS=coreml -DTorch_DIR=${Torch_DIR} ..
  make -j4 && make install
  ```

Please check [cmake build option](cmake_option.md).

### Install Model Converter

```bash
# You should use `conda install` to install the grpcio in requirements/runtime.txt
conda install grpcio
```

```bash
cd ${MMDEPLOY_DIR}
mim install -v -e .
```

**Note**

- Some dependencies are optional. Simply running `pip install -e .` will only install the minimum runtime requirements.
  To use optional dependencies, install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -e .[optional]`).
  Valid keys for the extras field are: `all`, `tests`, `build`, `optional`.

### Build SDK and Demo

The following shows an example of building an SDK using Core ML as the inference engine.

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
