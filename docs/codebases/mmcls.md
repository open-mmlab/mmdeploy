## MMClassification Support

[MMClassification](https://github.com/open-mmlab/mmclassification) is an open-source image classification toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com) project.

### MMClassification installation tutorial

Please refer to [install.md](https://github.com/open-mmlab/mmclassification/blob/master/docs/install.md) for installation.

### List of MMClassification models supported by MMDeploy

| model        | OnnxRuntime | TensorRT | NCNN | PPLNN | OpenVINO | model config file(example)                                                            |
|:-------------|:-----------:|:--------:|:----:|:---:|:--------:|:--------------------------------------------------------------------------------------|
| ResNet       |      Y      |    Y     |  Y   |  Y  |    ?     | $MMCLS_DIR/configs/resnet/resnet18_b32x8_imagenet.py                                  |
| ResNeXt      |      Y      |    Y     |  Y   |  Y  |    ?     | $MMCLS_DIR/configs/resnext/resnext50_32x4d_b32x8_imagenet.py                          |
| SE-ResNet    |      Y      |    Y     |  Y   |  Y  |    ?     | $MMCLS_DIR/configs/seresnet/seresnet50_b32x8_imagenet.py                              |
| MobileNetV2  |      Y      |    Y     |  Y   |  Y  |    ?     | $MMCLS_DIR/configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py                        |
| ShuffleNetV1 |      Y      |    Y     |  N   |  Y  |    ?     | $MMCLS_DIR/configs/shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet.py |
| ShuffleNetV2 |      Y      |    Y     |  N   |  Y  |    ?     | $MMCLS_DIR/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py |

### Reminder

None

### FAQs

None
