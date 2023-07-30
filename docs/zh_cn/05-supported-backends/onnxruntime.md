# onnxruntime 支持情况

## ONNX Runtime 介绍

**ONNX Runtime** 是一个跨平台的推理和训练加速器，与许多流行的ML/DNN框架兼容。查看其[github](https://github.com/microsoft/onnxruntime)以获取更多信息。

## 安装

*请注意，目前Linux平台只支持 **onnxruntime>=1.8.1** 。*

### 安装ONNX Runtime python包

- CPU 版本

```bash
pip install onnxruntime==1.8.1 # 如果你想用cpu版本
```

- GPU 版本

```bash
pip install onnxruntime-gpu==1.8.1 # 如果你想用gpu版本
```

### 安装float16转换工具(可选)

如果你想用float16精度，请执行以下脚本安装工具:

```bash
pip install onnx onnxconverter-common
```

## 构建自定义算子

### 下载ONNXRuntime库

从ONNX Runtime[发布版本](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1)下载`onnxruntime-linux-*.tgz`库，并解压，将onnxruntime所在路径添加到`ONNXRUNTIME_DIR`环境变量，最后将lib路径添加到`LD_LIBRARY_PATH`环境变量中，操作如下：

- CPU 版本

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

- GPU 版本

X64 GPU:

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-gpu-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-gpu-1.8.1.tgz
cd onnxruntime-linux-x64-gpu-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

Arm GPU:

```bash
# Arm not have 1.8.1 version package
wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-aarch64-1.10.0.tgz

tar -zxvf onnxruntime-linux-aarch64-1.10.0.tgz
cd onnxruntime-linux-aarch64-1.10.0
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

### 在Linux上构建

- CPU 版本

```bash
cd ${MMDEPLOY_DIR} # 进入MMDeploy根目录
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_DEVICES='cpu' -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc) && make install
```

- GPU 版本

```bash
cd ${MMDEPLOY_DIR} # 进入MMDeploy根目录
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_DEVICES='cuda' -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc) && make install
```

## 如何转换模型

- 你可以按照教程[如何转换模型](../02-how-to-run/convert_model.md)的说明去做

## 如何添加新的自定义算子

## 提示

- 自定义算子不包含在ONNX Runtime[支持的算子列表](https://github.com/microsoft/onnxruntime/blob/master/docs/OperatorKernels.md)中。
- 自定义算子应该能够导出到ONNX。

#### 主要过程

以自定义操作符`roi_align`为例。

1. 在ONNX Runtime源目录`${MMDEPLOY_DIR}/csrc/backend_ops/onnxruntime/`中创建一个`roi_align`目录
2. 添加头文件和源文件到`roi_align`目录`${MMDEPLOY_DIR}/csrc/backend_ops/onnxruntime/roi_align/`
3. 将单元测试添加到`tests/test_ops/test_ops.py`中。
   查看[这里](../../../tests/test_ops/test_ops.py)的例子。

\**最后，欢迎发送为MMDeploy添加ONNX Runtime自定义算子的PR。* \*: nerd_face:

## 参考

- [如何将具有自定义op的Pytorch模型导出为ONNX并在ONNX Runtime运行](https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md)
- [如何在ONNX Runtime添加自定义算子/内核](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
