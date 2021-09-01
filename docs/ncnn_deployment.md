# How to Deploy ncnn Models in deploy_prototype

This tutorial is based on Linux systems like Ubuntu-16.04.

Before starting this tutorial, you should make sure that the prerequisites mentioned by `deploy_prototype/README.md` are prepared.

## Preparation

- Download VulkanTools for the compilation of ncnn.
    ```bash
    wget https://sdk.lunarg.com/sdk/download/1.2.176.1/linux/vulkansdk-linux-x86_64-1.2.176.1.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.2.176.1.tar.gz
    tar -xf vulkansdk-linux-x86_64-1.2.176.1.tar.gz
    export VULKAN_SDK=$(pwd)/1.2.176.1/x86_64

- Prepare ncnn Framework

    - Download ncnn source code of tag 20210507
        ```bash
        git clone -b 20210507 git@github.com:Tencent/ncnn.git
        ```
    - <font color=red>Make install</font> ncnn library
        ```bash
        cd ncnn
        mkdir build
        cmake -DNCNN_VULKAN=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON -DNCNN_PYTHON=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_BENCHMARK=ON -DNCNN_BUILD_TESTS=ON ..
        make install
        ```
    - Install pyncnn module
        ```bash
        cd ncnn/python
        pip install .
        ```

- Build ncnn backend ops of deploy_prototype
    ```bash
    cd deploy_prototype
    mkdir build
    cd build
    cmake -DBUILD_NCNN_OPS=ON -DNCNN_DIR=${PATH_TO_NCNN}/ncnn ..
    ```
    The `${PATH_TO_NCNN}` refers as the root directory of ncnn source code.
- Install mmdeploy module
    ```bash
    cd deploy_prototype
    python setup.py develop
    ```
    Or you will fail on
    ```
    No module named mmdeploy
    ```

## FAQs

1. When running ncnn models for inference with custom ops, it fails and shows the error message like:

    ```
    TypeError: register mm custom layers(): incompatible function arguments. The following argument types are supported:
        1.(ar0: ncnn:Net) -> int

    Invoked with: <ncnn.ncnn.Net object at 0x7f7fc4038bb0>
    ```
    This is because of the failure to bind ncnn C++ library to pyncnn. You should build pyncnn from C++ ncnn source code, but not by `pip install`

2. When run the tools/deploy.py, it fails:
    ```
    Undefined symbol: __cpu_model
    ```
    This is a bug of gcc-5, you should update to `gcc >= 6`

## Performance Test

### MMCls
This table shows the performance of mmclassification models deployed on ncnn.

Dataset: ImageNet `val` dataset.

| Model | Top-1(%) | Top-5(%) |
|-------|----------|----------|
| MobileNetV2| 71.86 (71.86) | 90.42 (90.42) |
| ResNet | 69.88 (70.07) | 89.34 (89.44) |
| ResNeXt | 78.61 (78.71) | 94.17 (94.12) |

The data in the parentheses is the inference result from pytorch.
(According to: [mmcls model_zoo docs](https://github.com/open-mmlab/mmclassification/blob/master/docs/model_zoo.md))
