# Win10 下预编译包的使用

- [Win10 下预编译包的使用](#win10-下预编译包的使用)
  - [模型转换](#模型转换)
    - [Prerequisites](#prerequisites)
    - [ONNX Example](#onnx-example)
    - [TensorRT Example](#tensorrt-example)
  - [模型推理](#模型推理)
    - [Backend Inference](#backend-inference)
      - [ONNXRuntime](#onnxruntime)
      - [TensorRT](#tensorrt)
    - [Python SDK](#python-sdk)
      - [ONNXRuntime](#onnxruntime-1)
      - [TensorRT](#tensorrt-1)
    - [C SDK](#c-sdk)
      - [ONNXRuntime](#onnxruntime-2)
      - [TensorRT](#tensorrt-2)

______________________________________________________________________

目前，MMDeploy 在 Windows 平台下提供 TensorRT 以及 ONNXRuntime 两种预编译包，可以从[这里](https://github.com/open-mmlab/mmdeploy/releases)获取。 本篇教程以 mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1.zip 和 mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0.zip 为例来说明预编译包的使用方法。预编译包的目录结构如下，其中dist文件夹为模型转换相关内容，sdk为模型推理相关内容。

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

## 模型转换

### Prerequisites

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following official instructions, e.g.

On GPU platforms:

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

On CPU platforms:

```
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

**Step 3.** Install MMCV

On GPU platforms:

```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
```

On CPU platforms:

```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html
```

**Step 4.** Download MMDeploy repo (just clone, no need to build)

```
git clone https://github.com/open-mmlab/mmdeploy.git
```

### ONNX Example

下面以Resnet分类模型来说明用法

**Step 0.** Install mmdeploy package

```
pip install .\mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\dist\mmdeploy-0.6.0-py38-none-win_amd64.whl
```

**Step 1.** Install mmclassification

```
git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
pip install -e .
```

**Step 2.** Download checkpoint from [here](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)

Like this [ckpt](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth) which is trained by this [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py)

**Step 3.** Convert the model

The file structure of my working directory

```
..
|-- mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0
|-- mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1
|-- mmclassification
|-- mmdeploy
`-- resnet18_8xb32_in1k_20210831-fbbb1da6.pth
```

The python code to convert the model.

```
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

The file structure of work_dir after you run the python code.

```
.\work_dir\
`-- onnx
    `-- resnet
        |-- deploy.json
        |-- detail.json
        |-- end2end.onnx
        `-- pipeline.json
```

### TensorRT Example

下面以Resnet分类模型来说明用法

**Step 0.** Install mmdeploy package

```
pip install .\mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\dist\mmdeploy-0.6.0-py38-none-win_amd64.whl
# you may add --force-reinstal if you already install mmdeploy previously
```

**Step 1.** Install mmclassification

Same as ONNX Example Step 1

**Step 2.** Download checkpoint from [here](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)

Same as ONNX Example Step 2

**Step 3.** Install third-party package and set environment

**Step 3.1.** Install CUDA Toolkit 11.1

You can download from [here](https://developer.nvidia.com/cuda-11.1.1-download-archive)

**Step 3.2.** Install TensorRT 8.2.3.0

```
# The file structure of tensorrt package should be

.\TensorRT-8.2.3.0\
|-- bin
|-- data
|-- doc
|-- graphsurgeon
|-- include
|-- lib
|-- onnx_graphsurgeon
|-- python
|-- samples
`-- uff

# Install python package
pip install .\TensorRT-8.2.3.0\python\tensorrt-8.2.3.0-cp38-none-win_amd64.whl

# 设置环境变量 (系统属性-高级-环境变量)

在用户的Path变量中添加TensorRT的lib路径，具体位置根据实际调整，我这里是：
C:\Deps\tensorrt\TensorRT-8.2.3.0\lib

重启powershell让环境变量生效，可以通过 echo $env:PATH 来检查是否设置成功

如果你对环境变量不是很了解的话，建议只添加一个版本的TensorRT的lib到PATH里面。不建议拷贝TensorRT的dll到C盘的cuda目录，在某些情况下，这样可以暴露dll的版本问题。
```

**Step 3.3.** Install cuDNN 8.2.1.0

```
# The file structure of cudnn package should be
|-- NVIDIA_SLA_cuDNN_Support.txt
|-- bin
|-- include
`-- lib

# cuDNN不需要安装，只需要解压，并添加bin目录到环境变量就好，我这里是：
C:\Deps\cudnn\8.2.1\bin

注意事项同TensorRT

```

**Step 3.4.** Install pycuda

```
pip install pycuda
```

**Step 4.** Convert the model

The file structure of working directory

```
..
|-- mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0
|-- mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1
|-- mmclassification
|-- mmdeploy
`-- resnet18_8xb32_in1k_20210831-fbbb1da6.pth
```

The python code to convert the model.

```
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

The file structure of work_dir after you run the python code.

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

## 模型推理

以下内容假定已完成了上述模型转换的两个Example，并得到了上述展示的两个文件夹：

```
.\work_dir\onnx\resnet
.\work_dir\trt\resnet
```

### Backend Inference

这个接口不是为了做部署的，是为了用来检验转换的模型是否可以正常推理的。

#### ONNXRuntime

**Step 0.** Install ONNXRuntime

**Step 0.1.** 安装onnxruntime的python包

```
pip install onnxruntime==1.8.1
```

**Step 0.1.** 下载[onnxruntime](https://github.com/microsoft/onnxruntime/releases)，添加环境变量(这里是为了使用自定义算子)

```
# The file structure of onnxruntime should be
.\onnxruntime-win-gpu-x64-1.8.1\
|-- CodeSignSummary-c0f52e3d-f27b-4c42-a587-c8479a41573c.md
|-- GIT_COMMIT_ID
|-- LICENSE
|-- Privacy.md
|-- README.md
|-- ThirdPartyNotices.txt
|-- VERSION_NUMBER
|-- include
`-- lib

# 将lib目录添加到PATH里面，注意事项同TensorRT
```

**Step 1.** Inference

Current working directory

```
.
|-- mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0
|-- mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1
|-- mmclassification
|-- mmdeploy
|-- resnet18_8xb32_in1k_20210831-fbbb1da6.pth
`-- work_dir
```

Python code

```
from mmdeploy.apis import inference_model

model_cfg = 'mmclassification/configs/resnet/resnet18_8xb32_in1k.py'
deploy_cfg = 'mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py'
backend_files = ['work_dir/onnx/resnet/end2end.onnx']
img = 'mmclassification/demo/demo.JPEG'
device = 'cpu'
result = inference_model(model_cfg, deploy_cfg, backend_files, img, device)
```

**可能出现的问题：**

```
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Failed to load library, error code: 193
```

**原因：** 在较新的windows系统中，系统路径下下有两个`onnxruntime.dll`，且会优先加载，造成冲突。

```
C:\Windows\SysWOW64\onnxruntime.dll
C:\Windows\System32\onnxruntime.dll
```

**解决方法：** 以下两个方案任选其一

1. 将系统路径下的这两个dll改名，使其加载不到，可能涉及到修改文件权限的操作
2. 从[Github](<>)下载对应的版本，并将其中lib目录下的dll拷贝到mmdeploy_onnxruntime_ops.dll的同级目录
   （推荐使用Everything 进行查找，我这里是C:\\Software\\miniconda3\\envs\\openmmlab\\Lib\\site-packages\\mmdeploy\\lib\\mmdeploy_onnxruntime_ops.dll）

#### TensorRT

**Step 0.** 配置环境

按照上述转模型的要求，安装好CUDA Toolkit 11.1，TensorRT 8.2.3.0，cuDNN 8.2.1.0 并设置好环境变量

**Step 1.** Inference

Python code

```
from mmdeploy.apis import inference_model

model_cfg = 'mmclassification/configs/resnet/resnet18_8xb32_in1k.py'
deploy_cfg = 'mmdeploy/configs/mmcls/classification_tensorrt_static-224x224.py'
backend_files = ['work_dir/trt/resnet/end2end.engine']
img = 'mmclassification/demo/demo.JPEG'
device = 'cuda'
result = inference_model(model_cfg, deploy_cfg, backend_files, img, device)
```

### Python SDK

这里介绍如何使用SDK的Python API进行推理

#### ONNXRuntime

**Step 0.** 安装mmdeploy_python

```
pip install .\mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\sdk\python\mmdeploy_python-0.6.0-cp38-none-win_amd64.whl
```

**Step 1.** 下载[onnxruntime](https://github.com/microsoft/onnxruntime/releases)，添加环境变量

**Step 2.** Inference

```
python .\mmdeploy\demo\python\image_classification.py .\work_dir\onnx\resnet\ .\mmclassification\demo\demo.JPEG
```

#### TensorRT

**Step 0.** 安装mmdeploy_python

```
pip install .\mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\sdk\python\mmdeploy_python-0.6.0-cp38-none-win_amd64.whl
```

**Step 1.** 按照上述转模型的要求，安装好CUDA Toolkit 11.1，TensorRT 8.2.3.0，cuDNN 8.2.1.0 并设置好环境变量

**Step 2.** Inference

```
 python .\mmdeploy\demo\python\image_classification.py .\work_dir\trt\resnet\ .\mmclassification\demo\demo.JPEG --device-name cuda
```

### C SDK

这里介绍如何使用SDK的C API进行推理

这里涉及到编译，需要安装Vs2019+，CMake

example中读取图片用到了OpenCV，所以需要从这里安装[opencv-4.6.0-vc14_vc15.exe](https://github.com/opencv/opencv/releases)，或者自行编译

```
// Current working directories
.
|-- opencv
|-- mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0
|-- mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1
|-- mmclassification
|-- mmdeploy
|-- resnet18_8xb32_in1k_20210831-fbbb1da6.pth
`-- work_dir
```

#### ONNXRuntime

**Step 0.** 编译

在mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\\sdk\\example目录下

```
// 部分路径根据所在硬盘的位置进行修改
mkdir build
cd build
cmake .. -A x64 -T v142 `
  -DOpenCV_DIR=C:\workspace\opencv\build\x64\vc15\lib `
  -DMMDeploy_DIR=C:\workspace\mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\sdk\lib\cmake\MMDeploy `
  -DONNXRUNTIME_DIR=C:\Deps\onnxruntime\onnxruntime-win-gpu-x64-1.8.1

cmake --build . --config Release

```

**Step 1.** 添加环境变量

需要添加的环境变量有三个：分别是OpenCV的bin目录，mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\\sdk\\bin，以及onnxruntime的lib目录。在之前已经添加过的变量可以忽略，我这里是：

```
C:\Deps\onnxruntime\onnxruntime-win-gpu-x64-1.8.1\lib
C:\workspace\mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\sdk\bin
C:\workspace\opencv\build\x64\vc15\bin
```

这里也可以不添加环境变量，而将这三者的dll拷贝到刚才编译出的exe(build/Release)的同级目录下。

**Step 2.** Inference
重启Powershell让环境变量生效。这里建议使用cmd，这样如果exe运行时如果找不到相关的dll的话会有弹窗
在mmdeploy-0.6.0-windows-amd64-onnxruntime1.8.1\\sdk\\example\\build\\Release目录下：

```
.\image_classification.exe cpu C:\workspace\work_dir\onnx\resnet\ C:\workspace\mmclassification\demo\demo.JPEG
```

#### TensorRT

**Step 0.** 编译

在mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\\sdk\\example目录下

```
// 部分路径根据所在硬盘的位置进行修改
mkdir build
cd build
cmake .. -A x64 -T v142 `
  -DOpenCV_DIR=C:\workspace\opencv\build\x64\vc15\lib `
  -DMMDeploy_DIR=C:\workspace\mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\sdk\lib\cmake\MMDeploy `
  -DTENSORRT_DIR=C:\Deps\tensorrt\TensorRT-8.2.3.0 `
  -DCUDNN_DIR=C:\Deps\cudnn\8.2.1

cmake --build . --config Release
```

**可能出现的问题：**

```
enable_language(CUDA) 报错

-- Selecting Windows SDK version 10.0.19041.0 to target Windows 10.0.19044.
-- Found CUDA: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1 (found version "11.1")
CMake Error at C:/Software/cmake/cmake-3.23.1-windows-x86_64/share/cmake-3.23/Modules/CMakeDetermineCompilerId.cmake:491 (message):
  No CUDA toolset found.
Call Stack (most recent call first):
  C:/Software/cmake/cmake-3.23.1-windows-x86_64/share/cmake-3.23/Modules/CMakeDetermineCompilerId.cmake:6 (CMAKE_DETERMINE_COMPILER_ID_BUILD)
  C:/Software/cmake/cmake-3.23.1-windows-x86_64/share/cmake-3.23/Modules/CMakeDetermineCompilerId.cmake:59 (__determine_compiler_id_test)
  C:/Software/cmake/cmake-3.23.1-windows-x86_64/share/cmake-3.23/Modules/CMakeDetermineCUDACompiler.cmake:339 (CMAKE_DETERMINE_COMPILER_ID)
  C:/workspace/mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0/sdk/lib/cmake/MMDeploy/MMDeployConfig.cmake:27 (enable_language)
  CMakeLists.txt:5 (find_package)
```

**原因：** CUDA Toolkit 11.1安装在Visual Studio之前，造成VS的插件没有安装。或者VS的版本过新，使得CUDA Toolkit的安装的时候跳过了VS插件的安装

**解决方法：** 可以通过手工拷贝插件的方式来解决这个问题。
我这里的环境是CUDA Toolkit 11.1，vs2022，操作是将`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\extras\visual_studio_integration\MSBuildExtensions`中的四个文件拷贝到`C:\Software\Microsoft Visual Studio\2022\Community\Msbuild\Microsoft\VC\v170\BuildCustomizations` 目录下。具体路径根据实际情况进行更改。

**Step 1.** 添加环境变量
这里需要添加以下四个变量，根据各自的情况进行调整

```
C:\Deps\cudnn\8.2.1\bin
C:\Deps\tensorrt\TensorRT-8.2.3.0\lib
C:\workspace\opencv\build\x64\vc15\bin
C:\workspace\mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\sdk\bin
```

**Step 2.** Inference
重启Powershell让环境变量生效。这里建议使用cmd，这样如果exe运行时如果找不到相关的dll的话会有弹窗
在mmdeploy-0.6.0-windows-amd64-cuda11.1-tensorrt8.2.3.0\\sdk\\example\\build\\Release目录下：

```
.\image_classification.exe cuda C:\workspace\work_dir\trt\resnet C:\workspace\mmclassification\demo\demo.JPEG
```
