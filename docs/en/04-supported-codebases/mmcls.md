# MMClassification Deployment

[MMClassification](https://github.com/open-mmlab/mmclassification) aka `mmcls` is an open-source image classification toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com) project.

## Installation

### Install mmcls

If you have already done that, move on to the next section. Otherwise, please follow this [quick guide](https://github.com/open-mmlab/mmclassification/tree/1.x#installation) to finish mmcls installation.

### Install mmdeploy

There are several methods to install mmdeploy, among which you can choose an appropriate one according to your target platform and device.

**Method I:** Install precompiled package

> **TODO**. MMDeploy hasn't released based on dev-1.x branch.

**Method II:** Build using scripts

If your target platform is **Ubuntu 18.04 or later version**, we encourage you to run
[scripts](../01-how-to-build/build_from_script.md). For example, the following commands help install mmdeploy as well as inference engine - `ONNX Runtime` automatically.

```shell
git clone --recursive -b dev-1.x https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**Method III:** Build from source

If neither **I** nor **II** meets your requirements, [building mmdeploy from source](../01-how-to-build/build_from_source.md) is the last option.

## Convert model

You can use `tools/deploy.py` to convert mmcls models to the specified backend models. Its detailed usage can be learned from [here](https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/02-how-to-run/convert_model.md#usage).

The following shows an example about converting `resnet18` model to onnx model that can be inferred by ONNX Runtime.

```shell
cd mmdeploy

# download mmcls model
mim download mmcls --config resnet18_8xb32_in1k --dest .

# convert mmcls model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmcls/classification_onnxruntime_dynamic.py \
    resnet18_8xb32_in1k.py \
    resnet18_8xb32_in1k_20210831-fbbb1da6.pth \
    tests/data/tiger.jpeg \
    --work-dir mmdeploy_models/mmcls/ort \
    --device cpu \
    --show \
    --dump-info
```

It is crucial to specify the correct deployment config during model conversion. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/dev-1.x/configs/mmcls) of all supported backends for mmclassification. The config filename pattern is:

```
classification_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml and etc.
- **{precision}:** fp16, int8. When it's empty, it means fp32
- **{static | dynamic}:** static shape or dynamic shape
- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `resnet18` to other backend models by changing the deployment config file `mmclassification_onnxruntime_dynamic.py` to [others](https://github.com/open-mmlab/mmdeploy/tree/dev-1.x/configs/mmcls), e.g., converting to tensorrt model by `classification_tensorrt-fp16_dynamic-224x224-224x224.py`.

```{tip}
When converting mmcls models to tensorrt models, --device should be set to "cuda"
```

## Model Specification

Before moving on to model inference chapter, let us talk more about the converted model structure which is very important to do model inference.

The converted model locates in the working directory like `mmdeploy_models/mmcls/ort` in the previous example. It includes:

```
mmdeploy_models/mmcls/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- **deploy.json**: meta information of `end2end.onnx`
- **pipeline.json**: inference pipeline of mmdeploy SDK
- **detail.json**: conversion parameters

And the whole package **mmdeploy_models/mmcls/ort** is defined as **mmdeploy SDK model**. In other words, **mmdeploy SDK model** includes not only backend model but also inference meta information.

## Backend model inference

mmdeploy provides a unified API named as `inference_model` to do this job, making all inference backends API transparent to users.

Take the previous converted `end2end.onnx` model as an example,

```shell
from mmdeploy.apis import inference_model
result = inference_model(
  model_cfg='./resnet18_8xb32_in1k.py',
  deploy_cfg='configs/mmcls/classification_onnxruntime_dynamic.py',
  backend_files=['mmdeploy_models/mmcls/ort/end2end.onnx'],
  img='tests/data/tiger.jpeg',
  device='cpu')
print(result)
```

## SDK model inference

You can also perform SDK model inference like following,

```python
from mmdeploy_python import Classifier
import cv2

img = cv2.imread('tests/data/tiger.jpeg')

# create a classifier
classifier = Classifier(model_path='./mmdeploy_models/mmcls/ort', device_name='cpu', device_id=0)
# perform inference
result = classifier(img)
# show inference result
for label_id, score in result:
    print(label_id, score)
```

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/master/demo).

## Supported models

| Model                                                                                                      | TorchScript | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |
| :--------------------------------------------------------------------------------------------------------- | :---------: | :----------: | :------: | :--: | :---: | :------: |
| [ResNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)                        |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [ResNeXt](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)                      |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [SE-ResNet](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)                   |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [MobileNetV2](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2)             |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [ShuffleNetV1](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v1)           |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [ShuffleNetV2](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2)           |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |
| [VisionTransformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/vision_transformer) |      Y      |      Y       |    Y     |  Y   |   ?   |    Y     |
| [SwinTransformer](https://github.com/open-mmlab/mmclassification/tree/master/configs/swin_transformer)     |      Y      |      Y       |    Y     |  N   |   ?   |    N     |
