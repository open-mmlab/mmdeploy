## MMSegmentation Support

MMSegmentation is an open source object segmentation toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMSegmentation installation tutorial

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md) for installation.

### List of MMSegmentation models supported by MMDeploy

|    model   | OnnxRuntime |    TensorRT   | NCNN |  PPL  | OpenVino |  model config file(example)                                                               |
|:---------- | :---------: | :-----------: | :---:| :---: | :------: | :---------------------------------------------------------------------------------------  |
| FCN        |      Y      |       Y       |   Y  |   Y   |    ?     | $PATH_TO_MMSEG/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py                          |
| PSPNet     |      Y      |       Y       |   N  |   Y   |    ?     | $PATH_TO_MMSEG/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py                    |
| DeepLabV3  |      Y      |       Y       |   Y  |   Y   |    ?     | $PATH_TO_MMSEG/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py              |
| DeepLabV3+ |      Y      |       Y       |   Y  |   Y   |    ?     | $PATH_TO_MMSEG/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py      |

### Reminder

None

### FAQs

None
