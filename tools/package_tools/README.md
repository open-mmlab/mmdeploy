# Precompiled package

This document is going to describe the way to build MMDeploy package.

## Prerequisites

- Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

- Create conda environments for python 3.6, 3.7, 3.8, 3.9 and 3.10, respectively.

  ```shell
  for PYTHON_VERSION in 3.6 3.7 3.8 3.9 3.10
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
  pip install pyyaml packaging
  cd the/root/path/of/mmdeploy
  python tools/package_tools/generate_build_config.py --backend 'ort' \
            --system linux --build-mmdeploy --device cpu --build-sdk \
            --build-sdk-monolithic --build-sdk-python --sdk-dynamic-net \
            --output config.yml
  python tools/package_tools/mmdeploy_builder.py --config config.yml --output-dir pack
  ```

- On Windows platform, open `Anaconda Powershell Prompt` from the start menu and execute:

  ```shell
  conda activate mmdeploy-3.6
  pip install pyyaml packaging
  cd the/root/path/of/MMDeploy
  python tools/package_tools/generate_build_config.py --backend 'ort' \
            --system windows --build-mmdeploy --device cpu --build-sdk \
            --build-sdk-monolithic --build-sdk-python --sdk-dynamic-net \
            --output config.yml
  python tools/package_tools/mmdeploy_builder.py --config config.yml --output-dir pack
  ```
