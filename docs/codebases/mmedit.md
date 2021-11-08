# MMEditing Support

[MMEditing](https://github.com/open-mmlab/mmediting) is an open-source image and video editing toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## MMEditing installation tutorial

Please refer to [official installation guide](https://mmediting.readthedocs.io/en/latest/install.html#installation) to install the codebase.

## List of MMEditing models supported by MMDeploy

| Model  | Task             | ONNX Runtime | TensorRT | NCNN | PPL | OpenVINO | Model Config File (Example)                                              |
|:-------|:-----------------|:------------:|:--------:|:----:|:---:|:--------:|:-------------------------------------------------------------------------|
| SRCNN  | super-resolution |      Y       |    Y     |  N   |  Y  |    N     | $MMEDIT_DIR/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py       |
| ESRGAN | super-resolution |      Y       |    Y     |  N   |  Y  |    N     | $MMEDIT_DIR/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py |

## Reminder

None

## FAQs

1. Why the precision of SRCNN running in TensorRT is lower than in PyTorch?

    SRCNN uses bicubic to upsample images. TensorRT doesn't support bicubic operation. Therefore, we replace this operation with bilinear, which may lower the precision.
