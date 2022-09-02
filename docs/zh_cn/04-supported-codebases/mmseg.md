# mmseg 模型支持列表

mmseg 是一个基于 PyTorch 的开源对象分割工具箱，也是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

## 安装 mmseg

参照 [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation)。

## 支持列表

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

## 注意事项

- 所有 mmseg 模型仅支持 "whole" 推理模式。

- <i id=“static_shape”>PSPNet，Fast-SCNN</i> 仅支持静态输入，因为多数推理框架的 [nn.AdaptiveAvgPool2d](https://github.com/open-mmlab/mmsegmentation/blob/97f9670c5a4a2a3b4cfb411bcc26db16b23745f7/mmseg/models/decode_heads/psp_head.py#L38) 不支持动态输入。

- 对于仅支持静态形状的模型，应使用静态形状的部署配置文件，例如 `configs/mmseg/segmentation_tensorrt_static-1024x2048.py`
