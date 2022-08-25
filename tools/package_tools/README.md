# Precompiled package

This document is going to describe the way to build MMDeploy package.

## Prerequisites

- Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

- Create conda environments for python 3.6, 3.7, 3.8 and 3.9, respectively.

  ```shell
  for PYTHON_VERSION in 3.6 3.7 3.8 3.9
  do
    conda create --name mmdeploy-$PYTHON_VERSION python=$PYTHON_VERSION -y
  done
  ```

- Prepare MMDeploy dependencies

  Please follow the [build-on-Linux guide](../../docs/en/01-how-to-build/linux-x86_64.md) or [build-on-Windows guide](../../docs/en/01-how-to-build/linux-x86_64.md) to install dependencies of MMDeploy,
  including PyTorch, MMCV, OpenCV, ppl.cv, ONNX Runtime and TensorRT.

  Make sure the environment variables `pplcv_DIR`, `ONNXRUNTIME_DIR`, `TENSORRT_DIR`, `CUDNN_DIR` and `CUDA_TOOLKIT_ROOT_DIR` are exported.

## Run precompiled command

- On Linux platform,

  ```shell
  conda activate mmdeploy-3.6
  pip install pyyaml
  cd the/root/path/of/mmdeploy
  python tools/package_tools/mmdeploy_builder.py tools/package_tools/configs/linux_x64.yaml .
  ```

  You will get the precompiled packages `mmdeploy-{version}-linux-x86_64-cuda11.1-tensorrt8.2.3.0` and `mmdeploy-{version}-linux-x86_64-onnxruntime1.8.1` in the current directory if everything's going well.

- On Windows platform, open `Anaconda Powershell Prompt` from the start menu and execute:

  ```shell
  conda activate mmdeploy-3.6
  pip install pyyaml
  cd the/root/path/of/MMDeploy
  python tools/package_tools/mmdeploy_builder.py tools/package_tools/configs/windows_x64.yaml .
  ```

  When the build procedure finishes successfully, you will find `mmdeploy-{version}-windows-amd64-cuda11.1-tensorrt8.2.3.0` and `mmdeploy-{version}-windows-amd64-onnxruntime1.8.1` precompiled packages in the current directory.
