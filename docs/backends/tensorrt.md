## TensorRT Support

### Installation

#### Install TensorRT

Please install TensorRT 8 follow [install-guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

#### Build custom ops

Some custom ops are created to support models in OpenMMLab, the custom ops can be build as follow:

```bash
cd ${MMDEPLOY_DIR}
mkdir build
cd build
cmake -DBUILD_TENSORRT_OPS=ON ..
make -j$(nproc)
```

If you haven't install TensorRT in default path, Please add `-DTENSORRT_DIR` flag in cmake.

```bash
 cmake -DBUILD_TENSORRT_OPS=ON -DTENSORRT_DIR=${TENSORRT_DIR} ..
 make -j$(nproc)
```

### Convert model
