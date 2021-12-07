## ncnn Support

### Installation

#### Install ncnn

- Download VulkanTools for the compilation of ncnn.
    ```bash
    wget https://sdk.lunarg.com/sdk/download/1.2.176.1/linux/vulkansdk-linux-x86_64-1.2.176.1.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.2.176.1.tar.gz
    tar -xf vulkansdk-linux-x86_64-1.2.176.1.tar.gz
    export VULKAN_SDK=$(pwd)/1.2.176.1/x86_64
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
        git submodule update --init
        mkdir build
        cd build
        cmake -DNCNN_VULKAN=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON -DNCNN_PYTHON=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_BENCHMARK=ON -DNCNN_BUILD_TESTS=ON ..
        make install
        ```

    - Install pyncnn module
        ```bash
        cd ncnn/python
        pip install .
        ```

#### Build custom ops

Some custom ops are created to support models in OpenMMLab, the custom ops can be built as follows:

```bash
cd ${MMDEPLOY_DIR}
mkdir build
cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=ncnn ..
make -j$(nproc)
```

If you haven't installed NCNN in the default path, please add `-DNCNN_DIR` flag in cmake.

```bash
 cmake -DMMDEPLOY_TARGET_BACKENDS=ncnn -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn ..
 make -j$(nproc)
```

### Convert model

- This follows the tutorial on [How to convert model](../tutorials/how_to_convert_model.md).
- The converted model has two files: `.param` and `.bin`, as model structure file and weight file respectively.

### FAQs

1. When running ncnn models for inference with custom ops, it fails and shows the error message like:

    ```bash
    TypeError: register mm custom layers(): incompatible function arguments. The following argument types are supported:
        1.(ar0: ncnn:Net) -> int

    Invoked with: <ncnn.ncnn.Net object at 0x7f7fc4038bb0>
    ```

    This is because of the failure to bind ncnn C++ library to pyncnn. You should build pyncnn from C++ ncnn source code, but not by `pip install`
