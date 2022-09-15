# MMClassification Deployment

[MMClassification](https://github.com/open-mmlab/mmclassification) aka `mmcls` is an open-source image classification toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com) project.

## Installation

### Install mmcls

If you have already done that, move on to the next section. Otherwise, please follow this [quick guide](https://github.com/open-mmlab/mmclassification/tree/1.x#installation) to finish mmcls installation.

### Install mmdeploy

There are several methods to install mmdeploy, among which you can choose an appropriate one according to your target platform and device.

**Method I:** Install precompiled package

If your target device is NVIDIA GPU, it is recommended to follow [this](../01-how-to-build/install_prebuilt_package.md) guide to install the precompiled package and its corresponding inference engine - `TensorRT`.

<!-- The supported platform and device and inference engine matrix by precompiled package is presented as following:

| **Platform**   | **Device** | **Inference Engine** |
| -------------- | :--------: | :------------------: |
| Linux-x86_64   |    CPU     |     ONNX Runtime     |
| Linux-x86_64   |    CUDA    |       TensorRT       |
| Windows-x86_64 |    CPU     |     ONNX Runtime     |
| Windows-x86_64 |    CUDA    |       TensorRT       |
-->

**Method II:** Build using scripts

If your target platform is **Ubuntu 18.04 or later version**, we encourage you to run
[scripts](../01-how-to-build/build_from_script.md), by which mmdeploy as well as the inference engines, such as ONNX Runtime, ncnn, pplnn and torchscript, can be installed automatically.

**Method III:** Build from source

If neither **I** nor **II** meets your requirements, [building mmdeploy from source](../01-how-to-build/build_from_source.md) is the last option.

## Convert model

You can use `tools/deploy.py` to convert mmcls models to the specified backend models. Its detailed usage can be learned from [here](https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/02-how-to-run/convert_model.md#usage).

The following shows an example about converting `resnet18` model to onnx model that can be inferred by ONNX Runtime.

```shell
git clone --recursive https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy

# download mmcls model
pip install openmim
mim download mmcls --config resnet18_8xb16_cifar10 --dest .

# convert mmcls model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmcls/classification_onnxruntime_dynamic.py \
    resnet18_8xb16_cifar10.py \
    resnet18_b16x8_cifar10_20210528-bd6371c8.pth \
    tests/data/tiger.jpeg \
    --work-dir mmdeploy_models \
    --device cpu \
    --show \
    --dump-info
```

It is crucial to specify the correct deployment config during model conversion. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/master/configs/mmcls) of all supported backends for mmclassification. The config filename pattern is:

```
classification_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml and etc.
- **{precision}:** fp16, int8. When it's empty, it means fp32
- **{static | dynamic}:** static shape or dynamic shape
- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `resnet18` to other backend models by changing the deployment config file `mmclassification_onnxruntime_dynamic.py` to [others](https://github.com/open-mmlab/mmdeploy/tree/master/configs/mmcls), e.g., converting to tensorrt model by `classification_tensorrt-fp16_dynamic-224x224-224x224.py`.

```{tip}
When converting mmcls models to tensorrt models, --device should be set to "cuda"
```

## Inference model

After the installation, you can perform inference by mmdeploy inference SDK.

```python
from mmdeploy_python import Classifier
import cv2

img = cv2.imread('tests/data/tiger.jpeg')

# create a classifier
classifier = Classifier(model_path='./mmdeploy_models', device_name='cpu', device_id=0)
# perform inference
result = classifier(img)
# show inference result
indices = [i for i in range(len(bboxes))]
for label_id, score in result:
        print(label_id, score)
```

Besides python API, mmdeploy Inference SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn the usage from the [demos](https://github.com/open-mmlab/mmdeploy/tree/master/demo).

## Supported models

| Model             | TorchScript | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |                                          Model config                                           |
| :---------------- | :---------: | :----------: | :------: | :--: | :---: | :------: | :---------------------------------------------------------------------------------------------: |
| ResNet            |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)       |
| ResNeXt           |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)       |
| SE-ResNet         |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)      |
| MobileNetV2       |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |    [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2)    |
| ShuffleNetV1      |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |   [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v1)    |
| ShuffleNetV2      |      Y      |      Y       |    Y     |  Y   |   Y   |    Y     |   [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2)    |
| VisionTransformer |      Y      |      Y       |    Y     |  Y   |   ?   |    Y     | [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/vision_transformer) |
| SwinTransformer   |      Y      |      Y       |    Y     |  N   |   ?   |    N     |  [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/swin_transformer)  |
