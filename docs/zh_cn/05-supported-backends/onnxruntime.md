# onnxruntime 安装说明

## 介绍 onnxruntime

**onnxruntime** 是一个跨平台推理和训练加速器，与许多流行的 ML/DNN 框架兼容。[onnx github]（https://github.com/microsoft/onnxruntime）可提供更多信息。

## 安装

*对于 Linux 平台，目前需要 **onnxruntime>=1.8.1** 的 CPU 版本*

```bash
pip install onnxruntime==1.8.1
```

## 编译自定义算子

### 依赖

下载并设置环境变量

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

### 编译

```bash
cd ${MMDEPLOY_DIR} # To MMDeploy root directory
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc)
```

## How to convert a model

- You could follow the instructions of tutorial [How to convert model](../02-how-to-run/convert_model.md)

## How to add a new custom op

## Reminder

- The custom operator is not included in [supported operator list](https://github.com/microsoft/onnxruntime/blob/master/docs/OperatorKernels.md) in ONNX Runtime.
- The custom operator should be able to be exported to ONNX.

#### Main procedures

Take custom operator `roi_align` for example.

1. Create a `roi_align` directory in ONNX Runtime source directory `${MMDEPLOY_DIR}/csrc/backend_ops/onnxruntime/`
2. Add header and source file into `roi_align` directory `${MMDEPLOY_DIR}/csrc/backend_ops/onnxruntime/roi_align/`
3. Add unit test into `tests/test_ops/test_ops.py`
   Check [here](../../../tests/test_ops/test_ops.py) for examples.

**Finally, welcome to send us PR of adding custom operators for ONNX Runtime in MMDeploy.** :nerd_face:

## References

- [How to export Pytorch model with custom op to ONNX and run it in ONNX Runtime](https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md)
- [How to add a custom operator/kernel in ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/master/docs/AddingCustomOp.md)
