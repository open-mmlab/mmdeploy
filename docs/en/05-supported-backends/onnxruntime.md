# onnxruntime Support

## Introduction of ONNX Runtime

**ONNX Runtime** is a cross-platform inference and training accelerator compatible with many popular ML/DNN frameworks. Check its [github](https://github.com/microsoft/onnxruntime) for more information.

## Installation

*Please note that only **onnxruntime>=1.8.1** of on Linux platform is supported by now.*

### Install ONNX Runtime python package

- CPU Version

```bash
pip install onnxruntime==1.8.1 # if you want to use cpu version
```

- GPU Version

```bash
pip install onnxruntime-gpu==1.8.1 # if you want to use gpu version
```

### Install float16 conversion tool (optional)

If you want to use float16 precision, install the tool by running the following script:

```bash
pip install onnx onnxconverter-common
```

## Build custom ops

### Download ONNXRuntime Library

Download `onnxruntime-linux-*.tgz` library from ONNX Runtime [releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1), extract it, expose `ONNXRUNTIME_DIR` and finally add the lib path to `LD_LIBRARY_PATH` as below:

- CPU Version

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

- GPU Version

In X64 GPU:

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-gpu-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-gpu-1.8.1.tgz
cd onnxruntime-linux-x64-gpu-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

In Arm GPU:

```bash
# Arm not have 1.8.1 version package
wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-aarch64-1.10.0.tgz

tar -zxvf onnxruntime-linux-aarch64-1.10.0.tgz
cd onnxruntime-linux-aarch64-1.10.0
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

You can also go to [ONNX Runtime Release](https://github.com/microsoft/onnxruntime/releases) to find corresponding release version package.

### Build on Linux

- CPU Version

```bash
cd ${MMDEPLOY_DIR} # To MMDeploy root directory
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_DEVICES='cpu' -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc) && make install
```

- GPU Version

```bash
cd ${MMDEPLOY_DIR} # To MMDeploy root directory
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_DEVICES='cuda' -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc) && make install
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
- [How to add a custom operator/kernel in ONNX Runtime](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
