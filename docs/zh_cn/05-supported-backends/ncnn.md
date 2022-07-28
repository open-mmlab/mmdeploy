# ncnn 推理

## 安装

### 安装 ncnn

- 下载 VulkanTools

  ```bash
  wget https://sdk.lunarg.com/sdk/download/1.2.176.1/linux/vulkansdk-linux-x86_64-1.2.176.1.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.2.176.1.tar.gz
  tar -xf vulkansdk-linux-x86_64-1.2.176.1.tar.gz
  export VULKAN_SDK=$(pwd)/1.2.176.1/x86_64
  export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
  ```

- 请确保 `gcc >= 6`.

- 安装 Protobuf

  ```bash
  apt-get install libprotobuf-dev protobuf-compiler
  ```

- 安装 ncnn

  - 下载

    ```bash
    git clone -b 20220216 git@github.com:Tencent/ncnn.git
    ```

  - <font color=red> make install</font> ncnn

    ```bash
    cd ncnn
    export NCNN_DIR=$(pwd)
    git submodule update --init
    mkdir -p build && cd build
    cmake -DNCNN_VULKAN=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON -DNCNN_PYTHON=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_BENCHMARK=ON -DNCNN_BUILD_TESTS=ON ..
    make install
    ```

  - 安装 pyncnn

    ```bash
    cd ${NCNN_DIR} # To ncnn root directory
    cd python
    pip install -e .
    ```

### 编译自定义算子

为了更好地支持 OpenMMlab 相关算法，mmdeploy 增加了一些自定义算子 

```bash
cd ${MMDEPLOY_DIR}
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=ncnn ..
make -j$(nproc)
```

如果 ncnn 没被安装进系统路径，在 cmake 中用 `-Dncnn_DIR` 指示安装位置。

```bash
 cmake -DMMDEPLOY_TARGET_BACKENDS=ncnn -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn ..
 make -j$(nproc)
```

## 模型转换

- 参照 [How to convert model](../02-how-to-run/convert_model.md).
- 转换后会生成 `.param` 和 `.bin` 两种文件，分别代表网络结构和权重

## 注意事项

* ncnn 的维度不能超过 4
* 如果出现了类似错误：

   ```bash
   TypeError: register mm custom layers(): incompatible function arguments. The following argument types are supported:
       1.(ar0: ncnn:Net) -> int

   Invoked with: <ncnn.ncnn.Net object at 0x7f7fc4038bb0>
   ```

   这是因为 pyncnn 和 c++ ncnn 版本不匹配，请保证 pyncnn 来自源码编译而非 `pip install`
