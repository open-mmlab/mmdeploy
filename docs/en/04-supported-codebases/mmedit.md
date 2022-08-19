# MMEditing Support

[MMEditing](https://github.com/open-mmlab/mmediting) is an open-source image and video editing toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## MMEditing installation tutorial

Please refer to [official installation guide](https://mmediting.readthedocs.io/en/latest/install.html#installation) to install the codebase.

## MMEditing models support

| Model       | Task             | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |                                          Model config                                          |
| :---------- | :--------------- | :----------: | :------: | :--: | :---: | :------: | :--------------------------------------------------------------------------------------------: |
| SRCNN       | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |     [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srcnn)      |
| ESRGAN      | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |     [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/esrgan)     |
| ESRGAN-PSNR | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |     [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/esrgan)     |
| SRGAN       | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     | [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan) |
| SRResNet    | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     | [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan) |
| Real-ESRGAN | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |  [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/real_esrgan)   |
| EDSR        | super-resolution |      Y       |    Y     |  Y   |   N   |    Y     |      [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/edsr)      |
| RDN         | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/rdn)       |
