# MMEditing Support

[MMEditing](https://github.com/open-mmlab/mmediting) is an open-source image and video editing toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## MMEditing installation tutorial

Please refer to [official installation guide](https://mmediting.readthedocs.io/en/latest/install.html#installation) to install the codebase.

## List of MMEditing models supported by MMDeploy

| Model  |                 Model Config File (Example)                  | ONNX Runtime | TensorRT | NCNN  |  PPL  |
| :----: | :----------------------------------------------------------: | :----------: | :------: | :---: | :---: |
| SRCNN  |    configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py    |      Y       |    Y     |   N   |   Y   |
| ESRGAN | configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py |      Y       |    Y     |   N   |   Y   |

## MMEditing deployment task type

| codebase |       task       |
| :------: | :--------------: |
|  mmedit  | super-resolution |

## Reminder

None

## FAQs

1. Why the precision of SRCNN running in TensorRT is lower than in PyTorch?

    SRCNN uses bicubic to upsample images. TensorRT doesn't support bicubic operation. Therefore, we replace this operation with bilinear, which may lower the precision.
