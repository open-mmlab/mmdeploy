# MMPretrain Deployment

- [MMPretrain Deployment](#mmpretrain-deployment)
  - [Installation](#installation)
    - [Install mmpretrain](#install-mmpretrain)
    - [Install mmdeploy](#install-mmdeploy)
  - [Convert model](#convert-model)
  - [Model Specification](#model-specification)
  - [Model inference](#model-inference)
    - [Backend model inference](#backend-model-inference)
    - [SDK model inference](#sdk-model-inference)
  - [Supported models](#supported-models)

______________________________________________________________________

[MMPretrain](https://github.com/open-mmlab/mmpretrain) aka `mmpretrain` is an open-source image classification toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com) project.

## Installation

### Install mmpretrain

Please follow this [quick guide](https://github.com/open-mmlab/mmpretrain/tree/main#installation) to install mmpretrain.

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

**Method III:** Build from source

If neither **I** nor **II** meets your requirements, [building mmdeploy from source](../01-how-to-build/build_from_source.md) is the last option.

## Convert model

You can use [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/tree/main/tools/deploy.py) to convert mmpretrain models to the specified backend models. Its detailed usage can be learned from [here](https://github.com/open-mmlab/mmdeploy/tree/main/docs/en/02-how-to-run/convert_model.md#usage).

The command below shows an example about converting `resnet18` model to onnx model that can be inferred by ONNX Runtime.

```shell
cd mmdeploy

# download resnet18 model from mmpretrain model zoo
mim download mmpretrain --config resnet18_8xb32_in1k --dest .

# convert mmpretrain model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmpretrain/classification_onnxruntime_dynamic.py \
    resnet18_8xb32_in1k.py \
    resnet18_8xb32_in1k_20210831-fbbb1da6.pth \
    tests/data/tiger.jpeg \
    --work-dir mmdeploy_models/mmpretrain/ort \
    --device cpu \
    --show \
    --dump-info
```

It is crucial to specify the correct deployment config during model conversion. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmpretrain) of all supported backends for mmpretrain. The config filename pattern is:

```
classification_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml and etc.
- **{precision}:** fp16, int8. When it's empty, it means fp32
- **{static | dynamic}:** static shape or dynamic shape
- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `resnet18` to other backend models by changing the deployment config file `classification_onnxruntime_dynamic.py` to [others](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmpretrain), e.g., converting to tensorrt-fp16 model by `classification_tensorrt-fp16_dynamic-224x224-224x224.py`.

```{tip}
When converting mmpretrain models to tensorrt models, --device should be set to "cuda"
```

## Model Specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmpretrain/ort` in the previous example. It includes:

```
mmdeploy_models/mmpretrain/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- \***.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmpretrain/ort** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmpretrain/classification_onnxruntime_dynamic.py'
model_cfg = './resnet18_8xb32_in1k.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmpretrain/ort/end2end.onnx']
image = 'tests/data/tiger.jpeg'

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
    output_file='output_classification.png')
```

### SDK model inference

You can also perform SDK model inference like following,

```python
from mmdeploy_runtime import Classifier
import cv2

img = cv2.imread('tests/data/tiger.jpeg')

# create a classifier
classifier = Classifier(model_path='./mmdeploy_models/mmpretrain/ort', device_name='cpu', device_id=0)
# perform inference
result = classifier(img)
# show inference result
for label_id, score in result:
    print(label_id, score)
```

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/main/demo).

## Supported models

| Model                                                                                              | TorchScript | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |
| :------------------------------------------------------------------------------------------------- | :---------: | :----------: | :------: | :--: | :---: | :------: |
| [ResNet](https://github.com/open-mmlab/mmpretrain/tree/main/configs/resnet)                        |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [ResNeXt](https://github.com/open-mmlab/mmpretrain/tree/main/configs/resnext)                      |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [SE-ResNet](https://github.com/open-mmlab/mmpretrain/tree/main/configs/seresnet)                   |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [MobileNetV2](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mobilenet_v2)             |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [MobileNetV3](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mobilenet_v3)             |      Y      |      Y       |    Y     |  Y   |   ?   |    Y     |
| [ShuffleNetV1](https://github.com/open-mmlab/mmpretrain/tree/main/configs/shufflenet_v1)           |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [ShuffleNetV2](https://github.com/open-mmlab/mmpretrain/tree/main/configs/shufflenet_v2)           |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [VisionTransformer](https://github.com/open-mmlab/mmpretrain/tree/main/configs/vision_transformer) |      Y      |      Y       |    Y     |  Y   |   ?   |    Y     |
| [SwinTransformer](https://github.com/open-mmlab/mmpretrain/tree/main/configs/swin_transformer)     |      Y      |      Y       |    Y     |  N   |   ?   |    Y     |
| [MobileOne](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mobileone)                  |      Y      |      Y       |    Y     |  Y   |   ?   |    Y     |
| [EfficientNet](https://github.com/open-mmlab/mmpretrain/tree/main/configs/efficientnet)            |      Y      |      Y       |    Y     |  N   |   ?   |    Y     |
| [Conformer](https://github.com/open-mmlab/mmpretrain/tree/main/configs/conformer)                  |      Y      |      Y       |    Y     |  N   |   ?   |    Y     |
| [EfficientFormer](https://github.com/open-mmlab/mmpretrain/tree/main/configs/efficientformer)      |      Y      |      Y       |    Y     |  N   |   ?   |    Y     |
