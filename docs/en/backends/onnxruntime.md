## ONNX Runtime Support

### Introduction of ONNX Runtime

**ONNX Runtime** is a cross-platform inferencing and training accelerator compatible with many popular ML/DNN frameworks. Check its [github](https://github.com/microsoft/onnxruntime) for more information.

### Installation

*Please note that only **onnxruntime>=1.8.1** of CPU version on Linux platform is supported by now.*

- Install ONNX Runtime python package

```bash
pip install onnxruntime==1.8.1
```

### Build custom ops

#### Prerequisite

- Download `onnxruntime-linux` from ONNX Runtime [releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1), extract it, expose `ONNXRUNTIME_DIR` and finally add the lib path to `LD_LIBRARY_PATH` as below:

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

Note:

- If you want to save onnxruntime env variables to bashrc, you could run

    ```bash
    echo '# set env for onnxruntime' >> ~/.bashrc
    echo "export ONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}" >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```

#### Build on Linux

```bash
cd ${MMDEPLOY_DIR} # To MMDeploy root directory
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc)
```

### How to convert a model

- You could follow the instructions of tutorial [How to convert model](../tutorials/how_to_convert_model.md)

### List of supported custom ops

| Operator                                                                     |  CPU  |  GPU  | MMDeploy Releases |
| :--------------------------------------------------------------------------- | :---: | :---: | :---------------- |
| [grid_sampler](../ops/onnxruntime.md#grid_sampler)                           |   Y   |   N   | master            |
| [MMCVModulatedDeformConv2d](../ops/onnxruntime.md#mmcvmodulateddeformconv2d) |   Y   |   N   | master            |

### How to add a new custom op

#### Reminder

- The custom operator is not included in [supported operator list](https://github.com/microsoft/onnxruntime/blob/master/docs/OperatorKernels.md) in ONNX Runtime.
- The custom operator should be able to be exported to ONNX.

#### Main procedures

Take custom operator `roi_align` for example.

1. Create a `roi_align` directory in ONNX Runtime source directory `${MMDEPLOY_DIR}/csrc/backend_ops/onnxruntime/`
2. Add header and source file into `roi_align` directory `${MMDEPLOY_DIR}/csrc/backend_ops/onnxruntime/roi_align/`
3. Add unit test into `tests/test_ops/test_ops.py`
   Check [here](../../../tests/test_ops/test_ops.py) for examples.

**Finally, welcome to send us PR of adding custom operators for ONNX Runtime in MMDeploy.** :nerd_face:

### FAQs

- None

### References

- [How to export Pytorch model with custom op to ONNX and run it in ONNX Runtime](https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md)
- [How to add a custom operator/kernel in ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/master/docs/AddingCustomOp.md)
