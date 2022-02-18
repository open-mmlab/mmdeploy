## MMSegmentation Support

MMSegmentation is an open source object segmentation toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMSegmentation installation tutorial

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation.

### List of MMSegmentation models supported by MMDeploy

| Model                       | OnnxRuntime | TensorRT | NCNN | PPLNN | OpenVino |                                       Model config                                       |
|:----------------------------|:-----------:|:--------:|:----:|:-----:|:--------:|:----------------------------------------------------------------------------------------:|
| FCN                         |      Y      |    Y     |  Y   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn)      |
| PSPNet[*](#static_shape)    |      Y      |    Y     |  Y   |   Y   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet)     |
| DeepLabV3                   |      Y      |    Y     |  Y   |   Y   |    Y     |   [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3)   |
| DeepLabV3+                  |      Y      |    Y     |  Y   |   Y   |    Y     | [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus) |
| Fast-SCNN[*](#static_shape) |      Y      |    Y     |  N   |   Y   |    Y     |   [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastscnn)    |
| UNet[*](#static_shape)      |      Y      |    Y     |  Y   |   Y   |    Y     |     [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/unet)      |
| ANN                         |      Y      |    Y     |  N   |   N   |    N     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| APCNet                      |      Y      |    Y     |  N   |   Y   |    N     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| BiSeNetV1                   |      Y      |    Y     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| BiSeNetV2                   |      Y      |    Y     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| CCNet                       |      N      |    N     |  N   |   N   |    N     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| CGNet                       |      Y      |    Y     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| DMNet                       |      Y      |    N     |  N   |   N   |    N     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| DNLNet                      |      Y      |    Y     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| EMANet                      |      N      |    N     |  N   |   N   |    N     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| EncNet                      |      Y      |    Y     |  N   |   N   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| ERFNet                      |      Y      |    Y     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| FastFCN                     |      Y      |    Y     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| GCNet                       |      Y      |    Y     |  N   |   N   |    N     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| ICNet                       |      N      |    N     |  N   |   N   |    N     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| ISANet                      |      Y      |    Y     |  N   |   N   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| NonLocal Net                |      Y      |    N     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| OCRNet                      |      Y      |    Y     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| PointRend                   |      Y      |    N     |  N   |   N   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| PSANet                      |      N      |    N     |  N   |   N   |    N     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| SegFormer                   |      N      |    N     |  N   |   N   |    N     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| Semantic FPN                |      Y      |    Y     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| STDC1                       |      Y      |    Y     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| STDC2                       |      Y      |    Y     |  N   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |
| UPerNet                     |      Y      |    Y     |  N   |   N   |    N     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/)        |


### Reminder

- Only `whole` inference mode is supported for all mmseg models.

- <i id="static_shape">PSPNet, Fast-SCNN</i> only support static shape, because [nn.AdaptiveAvgPool2d](https://github.com/open-mmlab/mmsegmentation/blob/97f9670c5a4a2a3b4cfb411bcc26db16b23745f7/mmseg/models/decode_heads/psp_head.py#L38) is not supported in most of backends dynamically.

- For models only supporting static shape, you should use the deployment config file of static shape such as `configs/mmseg/segmentation_tensorrt_static-1024x2048.py`.

### FAQs

None
