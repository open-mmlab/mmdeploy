# MMRotate Deployment

- [MMRotate Deployment](#mmrotate-deployment)
  - [Installation](#installation)
    - [Install mmrotate](#install-mmrotate)
    - [Install mmdeploy](#install-mmdeploy)
  - [Convert model](#convert-model)
  - [Model specification](#model-specification)
  - [Model inference](#model-inference)
    - [Backend model inference](#backend-model-inference)
    - [SDK model inference](#sdk-model-inference)
  - [Supported models](#supported-models)

______________________________________________________________________

[MMRotate](https://github.com/open-mmlab/mmrotate) is an open-source toolbox for rotated object detection based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## Installation

### Install mmrotate

Please follow the [installation guide](https://mmrotate.readthedocs.io/en/1.x/get_started.html) to install mmrotate.

### Install mmdeploy

There are several methods to install mmdeploy, among which you can choose an appropriate one according to your target platform and device.

**Method I:** Install precompiled package

You can refer to [get_started](https://mmdeploy.readthedocs.io/en/latest/get_started.html#installation)

**Method II:** Build using scripts

If your target platform is **Ubuntu 18.04 or later version**, we encourage you to run
[scripts](../01-how-to-build/build_from_script.md). For example, the following commands install mmdeploy as well as inference engine - `ONNX Runtime`.

```shell
git clone --recursive -b main https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**NOTE**:

- Adding `$(pwd)/build/lib` to `PYTHONPATH` is for importing mmdeploy SDK python module - `mmdeploy_runtime`, which will be presented in chapter [SDK model inference](#sdk-model-inference).
- When [inference onnx model by ONNX Runtime](#backend-model-inference), it requests ONNX Runtime library be found. Thus, we add it to `LD_LIBRARY_PATH`.

**Method III:** Build from source

If neither **I** nor **II** meets your requirements, [building mmdeploy from source](../01-how-to-build/build_from_source.md) is the last option.

## Convert model

You can use [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/blob/main/tools/deploy.py) to convert mmrotate models to the specified backend models. Its detailed usage can be learned from [here](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/02-how-to-run/convert_model.md#usage).

The command below shows an example about converting `rotated-faster-rcnn` model to onnx model that can be inferred by ONNX Runtime.

```shell
cd mmdeploy

# download rotated-faster-rcnn model from mmrotate model zoo
mim download mmrotate --config rotated-faster-rcnn-le90_r50_fpn_1x_dota --dest .
wget https://github.com/open-mmlab/mmrotate/raw/main/demo/dota_demo.jpg

# convert mmrotate model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmrotate/rotated-detection_onnxruntime_dynamic.py \
    rotated-faster-rcnn-le90_r50_fpn_1x_dota.py \
    rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth \
    dota_demo.jpg \
    --work-dir mmdeploy_models/mmrotate/ort \
    --device cpu \
    --show \
    --dump-info
```

It is crucial to specify the correct deployment config during model conversion. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmrotate) of all supported backends for mmrotate. The config filename pattern is:

```
rotated_detection-{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml etc.
- **{precision}:** fp16, int8. When it's empty, it means fp32
- **{static | dynamic}:** static shape or dynamic shape
- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `rotated-faster-rcnn` to other backend models by changing the deployment config file `rotated-detection_onnxruntime_dynamic` to [others](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmrotate), e.g., converting to tensorrt-fp16 model by `rotated-detection_tensorrt-fp16_dynamic-320x320-1024x1024.py`.

```{tip}
When converting mmrotate models to tensorrt models, --device should be set to "cuda"
```

## Model specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmrotate/ort` in the previous example. It includes:

```
mmdeploy_models/mmrotate/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- \***.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmrotate/ort** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model and visualize the results.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmrotate/rotated-detection_onnxruntime_dynamic.py'
model_cfg = './rotated-faster-rcnn-le90_r50_fpn_1x_dota.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmrotate/ort/end2end.onnx']
image = './dota_demo.jpg'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='./output.png')
```

### SDK model inference

You can also perform SDK model inference like following,

```python
from mmdeploy_runtime import RotatedDetector
import cv2
import numpy as np

img = cv2.imread('./dota_demo.jpg')

# create a detector
detector = RotatedDetector(model_path='./mmdeploy_models/mmrotate/ort', device_name='cpu', device_id=0)
# perform inference
det = detector(img)
```

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/main/demo).

## Supported models

| Model                                                                                             | OnnxRuntime | TensorRT |
| :------------------------------------------------------------------------------------------------ | :---------: | :------: |
| [Rotated RetinaNet](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/rotated_retinanet)    |      Y      |    Y     |
| [Rotated FasterRCNN](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/rotated_faster_rcnn) |      Y      |    Y     |
| [Oriented R-CNN](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/oriented_rcnn)           |      Y      |    Y     |
| [Gliding Vertex](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/gliding_vertex)          |      Y      |    Y     |
| [RTMDET-R](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/rotated_rtmdet)                |      Y      |    Y     |
