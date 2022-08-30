# mmcls 模型支持列表

[MMClassification](https://github.com/open-mmlab/mmclassification) 是基于 Python 的的图像分类工具，属于 [OpenMMLab](https://openmmlab.com)。

## 安装 mmcls

请参考 [install.md](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/install.md) 进行安装。

## 支持列表

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
