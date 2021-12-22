# MMEditing Support

[MMEditing](https://github.com/open-mmlab/mmediting) is an open-source image and video editing toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## MMEditing installation tutorial

Please refer to [official installation guide](https://mmediting.readthedocs.io/en/latest/install.html#installation) to install the codebase.

## MMEditing models support

| Model       | Task             | ONNX Runtime | TensorRT | NCNN  | PPLNN | OpenVINO | Model Config File                                                                            |
| :---------- | :--------------- | :----------: | :------: | :---: | :---: | :------: | :------------------------------------------------------------------------------------------- |
| SRCNN       | super-resolution |      Y       |    Y     |   Y   |   Y   |    Y     | $MMEDIT_DIR/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py                           |
| ESRGAN      | super-resolution |      Y       |    Y     |   Y   |   Y   |    Y     | $MMEDIT_DIR/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py                     |
| ESRGAN      | super-resolution |      Y       |    Y     |   Y   |   Y   |    Y     | $MMEDIT_DIR/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py               |
| SRGAN       | super-resolution |      Y       |    Y     |   Y   |   Y   |    Y     | $MMEDIT_DIR/configs/restorers/srresnet_srgan/srgan_x4c64b16_g1_1000k_div2k.py                |
| SRResNet    | super-resolution |      Y       |    Y     |   Y   |   Y   |    Y     | $MMEDIT_DIR/configs/restorers/srresnet_srgan/srgan_x4c64b16_g1_1000k_div2k.py                |
| Real-ESRGAN | super-resolution |      Y       |    Y     |   Y   |   Y   |    Y     | $MMEDIT_DIR/configs/restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost.py |
| EDSR        | super-resolution |      Y       |    Y     |   Y   |   N   |    Y     | $MMEDIT_DIR/configs/restorers/edsr/edsr_x2c64b16_g1_300k_div2k.py                            |
| EDSR        | super-resolution |      Y       |    Y     |   Y   |   N   |    Y     | $MMEDIT_DIR/configs/restorers/edsr/edsr_x3c64b16_g1_300k_div2k.py                            |
| EDSR        | super-resolution |      Y       |    Y     |   Y   |   N   |    Y     | $MMEDIT_DIR/configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k.py                            |
| RDN         | super-resolution |      Y       |    Y     |   Y   |   Y   |    Y     | $MMEDIT_DIR/configs/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k.py                             |
| RDN         | super-resolution |      Y       |    Y     |   Y   |   Y   |    Y     | $MMEDIT_DIR/configs/restorers/rdn/rdn_x3c64b16_g1_1000k_div2k.py                             |
| RDN         | super-resolution |      Y       |    Y     |   Y   |   Y   |    Y     | $MMEDIT_DIR/configs/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k.py                             |

## Reminder

None

## FAQs

None
