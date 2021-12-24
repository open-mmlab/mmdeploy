## MMSegmentation Support

MMSegmentation is an open source object segmentation toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMSegmentation installation tutorial

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation.

### List of MMSegmentation models supported by MMDeploy

| model                       | OnnxRuntime | TensorRT | NCNN  | PPLNN | OpenVino | model config file(example)                                                         |
| :-------------------------- | :---------: | :------: | :---: | :---: | :------: | :--------------------------------------------------------------------------------- |
| FCN                         |      Y      |    Y     |   Y   |   Y   |    Y     | ${MMSEG_DIR}/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py                     |
| PSPNet[*](#static_shape)    |      Y      |    Y     |   Y   |   Y   |    Y     | ${MMSEG_DIR}/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py               |
| DeepLabV3                   |      Y      |    Y     |   Y   |   Y   |    Y     | ${MMSEG_DIR}/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py         |
| DeepLabV3+                  |      Y      |    Y     |   Y   |   Y   |    Y     | ${MMSEG_DIR}/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py |
| Fast-SCNN[*](#static_shape) |      Y      |    Y     |   N   |   Y   |    Y     | ${MMSEG_DIR}/configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py              |

### Reminder

- Only `whole` inference mode is supported for all mmseg models.

- <i id="static_shape">PSPNet, Fast-SCNN</i> only support static shape, because [nn.AdaptiveAvgPool2d](https://github.com/open-mmlab/mmsegmentation/blob/97f9670c5a4a2a3b4cfb411bcc26db16b23745f7/mmseg/models/decode_heads/psp_head.py#L38) is not supported in most of backends dynamically.

- For models only supporting static shape, you should use the deployment config file of static shape such as `configs/mmseg/segmentation_tensorrt_static-1024x2048.py`.

### FAQs

None
