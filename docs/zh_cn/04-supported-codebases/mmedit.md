# mmedit 模型支持列表

[mmedit](https://github.com/open-mmlab/mmediting) 是基于 PyTorch 的开源图像和视频编辑工具箱，属于 [OpenMMLab](https://openmmlab.com/)。

## 安装 mmedit

参照 [official installation guide](https://mmediting.readthedocs.io/en/latest/install.html#installation)。

## 支持列表

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
