## ncnn Support

### Installation

#### Install ncnn

- Download VulkanTools for the compilation of ncnn.

    ```bash
    wget https://sdk.lunarg.com/sdk/download/1.2.176.1/linux/vulkansdk-linux-x86_64-1.2.176.1.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.2.176.1.tar.gz
    tar -xf vulkansdk-linux-x86_64-1.2.176.1.tar.gz
    export VULKAN_SDK=$(pwd)/1.2.176.1/x86_64
    export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
    ```

- Check your gcc version.
You should ensure your gcc satisfies `gcc >= 6`.

- Install Protocol Buffers through:
    ```bash
    apt-get install libprotobuf-dev protobuf-compiler
    ```

- Prepare ncnn Framework

    - Download ncnn source code
        ```bash
        git clone git@github.com:Tencent/ncnn.git
        ```

    - <font color=red>Make install</font> ncnn library
        ```bash
        cd ncnn
        export NCNN_DIR=$(pwd)
        git submodule update --init
        mkdir -p build && cd build
        cmake -DNCNN_VULKAN=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON -DNCNN_PYTHON=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_BENCHMARK=ON -DNCNN_BUILD_TESTS=ON ..
        make install
        ```

    - Install pyncnn module
        ```bash
        cd ${NCNN_DIR} # To NCNN root directory
        cd python
        pip install -e .
        ```

#### Build custom ops

Some custom ops are created to support models in OpenMMLab, the custom ops can be built as follows:

```bash
cd ${MMDEPLOY_DIR}
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=ncnn ..
make -j$(nproc)
```

If you haven't installed NCNN in the default path, please add `-Dncnn_DIR` flag in cmake.

```bash
 cmake -DMMDEPLOY_TARGET_BACKENDS=ncnn -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn ..
 make -j$(nproc)
```

### Convert model

- This follows the tutorial on [How to convert model](../tutorials/how_to_convert_model.md).
- The converted model has two files: `.param` and `.bin`, as model structure file and weight file respectively.


### List of supported custom ops

| Operator                        | CPU | MMDeploy Releases |
|:--------------------------------|:---:|:------------------|
| [Expand](../ops/ncnn.md#expand) |  Y  | master            |
| [Gather](../ops/ncnn.md#gather) |  Y  | master            |
| [Shape](../ops/ncnn.md#shape)   |  Y  | master            |
| [TopK](../ops/ncnn.md#topk)     |  Y  | master            |

#### Reminder

- If ncnn version >= 1.0.20201208, the dimension of ncnn.Mat should be no more than 4, or the dimension of the ncnn.Mat should be no more than 3.

### FAQs

1. When running ncnn models for inference with custom ops, it fails and shows the error message like:

    ```bash
    TypeError: register mm custom layers(): incompatible function arguments. The following argument types are supported:
        1.(ar0: ncnn:Net) -> int

    Invoked with: <ncnn.ncnn.Net object at 0x7f7fc4038bb0>
    ```

    This is because of the failure to bind ncnn C++ library to pyncnn. You should build pyncnn from C++ ncnn source code, but not by `pip install`
