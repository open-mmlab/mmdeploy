# MMSegmentation Support

MMSegmentation is an open source object segmentation toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## MMSegmentation installation tutorial

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation.

## List of MMSegmentation models supported by MMDeploy

| Model                        | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVino |                                       Model config                                       |
| :--------------------------- | :---------: | :------: | :--: | :---: | :------: | :--------------------------------------------------------------------------------------: |
| FCN                          |      Y      |    Y     |  Y   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn)      |
| PSPNet[\*](#static_shape)    |      Y      |    Y     |  Y   |   Y   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet)     |
| DeepLabV3                    |      Y      |    Y     |  Y   |   Y   |    Y     |   [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3)   |
| DeepLabV3+                   |      Y      |    Y     |  Y   |   Y   |    Y     | [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus) |
| Fast-SCNN[\*](#static_shape) |      Y      |    Y     |  N   |   Y   |    Y     |   [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastscnn)    |
| UNet                         |      Y      |    Y     |  Y   |   Y   |    Y     |     [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/unet)      |
| ANN[\*](#static_shape)       |      Y      |    Y     |  N   |   N   |    N     |      [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ann)      |
| APCNet                       |      Y      |    Y     |  Y   |   N   |    N     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/apcnet)     |
| BiSeNetV1                    |      Y      |    Y     |  Y   |   N   |    Y     |   [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/bisenetv1)   |
| BiSeNetV2                    |      Y      |    Y     |  Y   |   N   |    Y     |   [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/bisenetv2)   |
| CGNet                        |      Y      |    Y     |  Y   |   N   |    Y     |     [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/cgnet)     |
| DMNet                        |      Y      |    N     |  N   |   N   |    N     |     [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/dmnet)     |
| DNLNet                       |      Y      |    Y     |  Y   |   N   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/dnlnet)     |
| EMANet                       |      Y      |    Y     |  N   |   N   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/emanet)     |
| EncNet                       |      Y      |    Y     |  N   |   N   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/encnet)     |
| ERFNet                       |      Y      |    Y     |  Y   |   N   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/erfnet)     |
| FastFCN                      |      Y      |    Y     |  Y   |   N   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastfcn)    |
| GCNet                        |      Y      |    Y     |  N   |   N   |    N     |     [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/gcnet)     |
| ICNet[\*](#static_shape)     |      Y      |    Y     |  N   |   N   |    Y     |     [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/icnet)     |
| ISANet[\*](#static_shape)    |      Y      |    Y     |  N   |   N   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/isanet)     |
| NonLocal Net                 |      Y      |    Y     |  Y   |   N   |    Y     | [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/nonlocal_net)  |
| OCRNet                       |      Y      |    Y     |  Y   |   N   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ocrnet)     |
| PointRend[\*](#static_shape) |      Y      |    Y     |  N   |   N   |    N     |  [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/point_rend)   |
| Semantic FPN                 |      Y      |    Y     |  Y   |   N   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/sem_fpn)    |
| STDC                         |      Y      |    Y     |  Y   |   N   |    Y     |     [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/stdc)      |
| UPerNet[\*](#static_shape)   |      Y      |    Y     |  N   |   N   |    N     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/upernet)    |
| DANet                        |      Y      |    Y     |  N   |   N   |    Y     |     [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/danet)     |
| Segmenter[\*](#static_shape) |      Y      |    Y     |  Y   |   N   |    Y     |   [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segmenter)   |
| SegFormer[\*](#static_shape) |      Y      |    Y     |  N   |   N   |    Y     |   [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segformer)   |
| SETR                         |      Y      |    N     |  N   |   N   |    Y     |     [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/setr)      |
| CCNet                        |      N      |    N     |  N   |   N   |    N     |     [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ccnet)     |
| PSANet                       |      N      |    N     |  N   |   N   |    N     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/psanet)     |
| DPT                          |      N      |    N     |  N   |   N   |    N     |      [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/dpt)      |

## Reminder

- Only `whole` inference mode is supported for all mmseg models.

- <i id="static_shape">PSPNet, Fast-SCNN</i> only support static shape, because [nn.AdaptiveAvgPool2d](https://github.com/open-mmlab/mmsegmentation/blob/97f9670c5a4a2a3b4cfb411bcc26db16b23745f7/mmseg/models/decode_heads/psp_head.py#L38) is not supported in most of backends dynamically.

- For models only supporting static shape, you should use the deployment config file of static shape such as `configs/mmseg/segmentation_tensorrt_static-1024x2048.py`.
