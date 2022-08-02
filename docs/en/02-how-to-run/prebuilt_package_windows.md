# How to use prebuilt package on Windows10

- [How to use prebuilt package on Windows10](#how-to-use-prebuilt-package-on-windows10)
  - [Prerequisite](#prerequisite)
    - [ONNX Runtime](#onnx-runtime)
    - [TensorRT](#tensorrt)
  - [Model Convert](#model-convert)
    - [ONNX Runtime Example](#onnx-runtime-example)
    - [TensorRT Example](#tensorrt-example)
  - [Model Inference](#model-inference)
    - [Backend Inference](#backend-inference)
      - [ONNXRuntime](#onnxruntime)
      - [TensorRT](#tensorrt-1)
    - [Python SDK](#python-sdk)
      - [ONNXRuntime](#onnxruntime-1)
      - [TensorRT](#tensorrt-2)
    - [C SDK](#c-sdk)
      - [ONNXRuntime](#onnxruntime-2)
      - [TensorRT](#tensorrt-3)
  - [Possible problems](#possible-problems)

______________________________________________________________________

Currently, `MMDeploy` provides prebuilt packages with `TensorRT` or `ONNX Runtime` as inference backend on `Windows`. You can download from [Releases](https://github.com/open-mmlab/mmdeploy/releases) page.

This tutorial takes `mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1.zip` and `mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0.zip` as examples to show how to use the prebuilt packages.

In order to facilitate users to get started quickly, this tutorial takes the classification model (mmclassification) as an example to show the use of two prebuilt packages.

The directory structure of the prebuilt package is as follows, where the `dist` folder is the model conversion related content, and the `sdk` folder is the model inference related content.

```
.
|-- dist
`-- sdk
    |-- bin
    |-- example
    |-- include
    |-- lib
    `-- python
```

## Prerequisite

To use the prebuilt package to perform `model conversion` and `model inference`, you need to install some third-party dependent libraries. The following introduces preparations for using `ONNX Runtime` and `TensorRT` as the inference backend.

In the environment preparation of the two inference backend, some operations are common. The following describes these common operations first, and then introduces their own unique operations.

First create a new working directory workspace

Please follow the [get_started](../get_started.md) documentation to

1. Follow the [get_started](../get_started.md) documentation to create a virtual python environment and install pytorch, torchvision, and mmcv-full. To use the C interface of the SDK, you need to install vs2019+, OpenCV.

   :point_right: It is recommended to use `pip` instead of `conda` to install pytorch, torchvision

2. Clone the mmdeploy repository

   ```bash
   git clone https://github.com/open-mmlab/mmdeploy.git
   ```

   :point_right: The main purpose here is to use the configs, so there is no need to add `--recursive` to download the submodule. It's also no need to compile `mmdeploy`.

3. Install mmclassification

   ```bash
   git clone https://github.com/open-mmlab/mmclassification.git
   cd mmclassification
   pip install -e .
   ```

4. Prepare a PyTorch model as our example

   Download the pth [resnet18_8xb32_in1k_20210831-fbbb1da6.pth](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth). The corresponding config is [resnet18_8xb32_in1k.py](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py)

After doing the above work, the structure of the current working directory should be:

```
.
|-- mmclassification
|-- mmdeploy
|-- resnet18_8xb32_in1k_20210831-fbbb1da6.pth
```

### ONNX Runtime

This section describes the environment preparations specific to `mmdeploy` for using `ONNX Runtime` as inference backend.

Install precompiled packages of `mmdeploy` (model transformation) and `mmdeploy_python` (model inference Python API)

5. Install prebuilt package for `mmdeploy` (Model Convert) and `mmdeploy_python` (Model Inferece Python API).

   ```bash
   # download mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1.zip
   pip install .\mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\dist\mmdeploy-0.6.0-py38-none-win_amd64.whl
   pip install .\mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\sdk\python\mmdeploy_python-0.6.0-cp38-none-win_amd64.whl
   ```

   :point_right: If you have installed it before, please uninstall it first.

6. Install onnxruntime package

   ```
   pip install onnxruntime==1.8.1
   ```

7. Download [`onnxruntime`](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1), and add environment variable.

   Add the lib directory of onnxruntime to the PATH, as shown in the figure, the specific path should be changed according to individual circumstances.

   ![sys-path](https://user-images.githubusercontent.com/16019484/181463801-1d7814a8-b256-46e9-86f2-c08de0bc150b.png)
   :exclamation: Restart powershell to make the environment variables take effect. You can check whether the settings are in effect by echo $env:PATH.

### TensorRT

This section describes the environment preparations specific to `mmdeploy` for using `TensorRT` as inference backend.

5. Install prebuilt package for `mmdeploy` (Model Convert) and `mmdeploy_python` (Model Inferece Python API).

   ```bash
   # download mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0.zip
   pip install .\mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\dist\mmdeploy-0.6.0-py38-none-win_amd64.whl
   pip install .\mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\sdk\python\mmdeploy_python-0.6.0-cp38-none-win_amd64.whl
   ```

   :point_right: If you have installed it before, please uninstall it first.

6. Install CUDA related content and set environment variables

   - CUDA Toolkit 11.1
   - TensorRT 8.2.3.0 (python package + environment variables)
   - cuDNN 8.2.1.0

   The CUDA environment variable will be automatically added after installing the CUDA Toolkit. For TensorRT and cuDNN, after decompressing the zip package, you need to add the path of the runtime library to the PATH. You can refer to the path setting of onnxruntime.

   :exclamation: Restart powershell to make the environment variables take effect. You can check whether the settings are in effect by echo $env:PATH.

   :exclamation: It is recommended to add only one version of the TensorRT lib to the PATH. It is not recommended to copy the dll of TensorRT to the cuda directory in C drive. In some cases, this can expose the version problem when running the executable program.

7. Install pycuda by `pip install pycuda`

## Model Convert

### ONNX Runtime Example

The following describes how to use the `mmdeploy` prebuilt package to perform model conversion based on the previously downloaded ckpt.

After doing the above work, the structure of the current working directory should be：

```
..
|-- mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1
|-- mmclassification
|-- mmdeploy
`-- resnet18_8xb32_in1k_20210831-fbbb1da6.pth
```

The python to convert the model

```python
from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = 'mmclassification/demo/demo.JPEG'
work_dir = 'work_dir/onnx/resnet'
save_file = 'end2end.onnx'
deploy_cfg = 'mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py'
model_cfg = 'mmclassification/configs/resnet/resnet18_8xb32_in1k.py'
model_checkpoint = 'resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
device = 'cpu'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint)
```

The structure of the converted model directory:

```bash
.\work_dir\
`-- onnx
    `-- resnet
        |-- deploy.json
        |-- detail.json
        |-- end2end.onnx
        `-- pipeline.json
```

### TensorRT Example

The following describes how to use the `mmdeploy` prebuilt package to perform model conversion based on the previously downloaded ckpt.

After doing the above work, the structure of the current working directory should be：

```
..
|-- mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0
|-- mmclassification
|-- mmdeploy
`-- resnet18_8xb32_in1k_20210831-fbbb1da6.pth
```

The python to convert the model

```python
from mmdeploy.apis import torch2onnx
from mmdeploy.apis.tensorrt import onnx2tensorrt
from mmdeploy.backend.sdk.export_info import export2SDK
import os

img = 'mmclassification/demo/demo.JPEG'
work_dir = 'work_dir/trt/resnet'
save_file = 'end2end.onnx'
deploy_cfg = 'mmdeploy/configs/mmcls/classification_tensorrt_static-224x224.py'
model_cfg = 'mmclassification/configs/resnet/resnet18_8xb32_in1k.py'
model_checkpoint = 'resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
device = 'cpu'

# 1. convert model to IR(onnx)
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. convert IR to tensorrt
onnx_model = os.path.join(work_dir, save_file)
save_file = 'end2end.engine'
model_id = 0
device = 'cuda'
onnx2tensorrt(work_dir, save_file, model_id, deploy_cfg, onnx_model, device)

# 3. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint)
```

The structure of the converted model directory:

```
.\work_dir\
`-- trt
    `-- resnet
        |-- deploy.json
        |-- detail.json
        |-- end2end.engine
        |-- end2end.onnx
        `-- pipeline.json
```

## Model Inference

The following content assumes that the two examples of the above model conversion have been completed, and one or both of the two model folders after the above model conversion have been obtained:

```
.\work_dir\onnx\resnet
.\work_dir\trt\resnet
```

The structure of current working directory：

```
.
|-- mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0
|-- mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1
|-- mmclassification
|-- mmdeploy
|-- resnet18_8xb32_in1k_20210831-fbbb1da6.pth
`-- work_dir
```

### Backend Inference

:exclamation: It should be emphasized that this interface is not for deployment, but shields the difference of backend interface api. The main purpose of this api is to check whether the converted model can be inferred normally.

#### ONNXRuntime

The Python code to inference model

```python
from mmdeploy.apis import inference_model

model_cfg = 'mmclassification/configs/resnet/resnet18_8xb32_in1k.py'
deploy_cfg = 'mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py'
backend_files = ['work_dir/onnx/resnet/end2end.onnx']
img = 'mmclassification/demo/demo.JPEG'
device = 'cpu'
result = inference_model(model_cfg, deploy_cfg, backend_files, img, device)
```

#### TensorRT

The Python code to inference model

```python
from mmdeploy.apis import inference_model

model_cfg = 'mmclassification/configs/resnet/resnet18_8xb32_in1k.py'
deploy_cfg = 'mmdeploy/configs/mmcls/classification_tensorrt_static-224x224.py'
backend_files = ['work_dir/trt/resnet/end2end.engine']
img = 'mmclassification/demo/demo.JPEG'
device = 'cuda'
result = inference_model(model_cfg, deploy_cfg, backend_files, img, device)
```

### Python SDK

The following describes how to use the SDK's Python API for inference

#### ONNXRuntime

The inference code

```bash
python .\mmdeploy\demo\python\image_classification.py .\work_dir\onnx\resnet\ .\mmclassification\demo\demo.JPEG
```

#### TensorRT

The inference code

```
 python .\mmdeploy\demo\python\image_classification.py .\work_dir\trt\resnet\ .\mmclassification\demo\demo.JPEG --device-name cuda
```

### C SDK

The following describes how to use the SDK's C API for inference

#### ONNXRuntime

1. Build examples

   Under `mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\sdk\example` directory

   ```
   // Path should be modified according to the actual location
   mkdir build
   cd build
   cmake .. -A x64 -T v142 `
     -DOpenCV_DIR=C:\Deps\opencv\build\x64\vc15\lib `
     -DMMDeploy_DIR=C:\workspace\mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\sdk\lib\cmake\MMDeploy `
     -DONNXRUNTIME_DIR=C:\Deps\onnxruntime\onnxruntime-win-gpu-x64-1.8.1

   cmake --build . --config Release
   ```

2. Add environment variables or copy the runtime dll to the level directory of exe

   :point_right: The purpose is to make the exe find the relevant dll

   If choose to add environment variables, add the runtime library path of `mmdeploy` (`mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\sdk\bin`) to the PATH. You can refer to the process of adding onnxruntime lib path.

   If choose to copy the dynamic library, copy the dll in the bin directory to the same level directory of the just compiled exe (build/Release).

3. Inference：

   It is recommended to use cmd here. It would pop-up a window showing the missing dll if you didn't set the path right.

   Under `mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\\sdk\\example\\build\\Release` directory：

   ```
   .\image_classification.exe cpu C:\workspace\work_dir\onnx\resnet\ C:\workspace\mmclassification\demo\demo.JPEG
   ```

#### TensorRT

1. Build examples

   Under `mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\\sdk\\example` directory

   ```
   // Path should be modified according to the actual location
   mkdir build
   cd build
   cmake .. -A x64 -T v142 `
     -DOpenCV_DIR=C:\Deps\opencv\build\x64\vc15\lib `
     -DMMDeploy_DIR=C:\workspace\mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8 2.3.0\sdk\lib\cmake\MMDeploy `
     -DTENSORRT_DIR=C:\Deps\tensorrt\TensorRT-8.2.3.0 `
     -DCUDNN_DIR=C:\Deps\cudnn\8.2.1
   cmake --build . --config Release
   ```

2. Add environment variables or copy the runtime dll to the level directory of exe

   :point_right: The purpose is to make the exe find the relevant dll

   If choose to add environment variables, add the runtime library path of `mmdeploy` (`mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\sdk\bin`) to the PATH. You can refer to the process of adding onnxruntime lib path.

   If choose to copy the dynamic library, copy the dll in the bin directory to the same level directory of the just compiled exe (build/Release).

3. Inference

   It is recommended to use cmd here. It would pop-up a window showing the missing dll if you didn't set the path right.

   Under `mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\\sdk\\example\\build\\Release` directory

   ```
   .\image_classification.exe cuda C:\workspace\work_dir\trt\resnet C:\workspace\mmclassification\demo\demo.JPEG
   ```

## Possible problems

If you encounter problems, please refer to [FAQ](../faq.md)
