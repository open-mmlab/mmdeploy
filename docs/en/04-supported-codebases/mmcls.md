# MMClassification Support

[MMClassification](https://github.com/open-mmlab/mmclassification) is an open-source image classification toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com) project.

## MMClassification installation tutorial

Please refer to [install.md](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/install.md) for installation.

## List of MMClassification models supported by MMDeploy

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
