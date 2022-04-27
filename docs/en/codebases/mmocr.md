## MMOCR Support

MMOCR is an open-source toolbox based on PyTorch and mmdetection for text detection, text recognition, and the corresponding downstream tasks including key information extraction. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMOCR installation tutorial

Please refer to [install.md](https://mmocr.readthedocs.io/en/latest/install.html) for installation.

### List of MMOCR models supported by MMDeploy

| Model | Task             | OnnxRuntime | TensorRT | ncnn  | PPLNN | OpenVINO |                                  Model config                                  |
| :---- | :--------------- | :---------: | :------: | :---: | :---: | :------: | :----------------------------------------------------------------------------: |
| DBNet | text-detection   |      Y      |    Y     |   Y   |   Y   |    Y     | [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/dbnet)  |
| CRNN  | text-recognition |      Y      |    Y     |   Y   |   Y   |    N     | [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/crnn) |
| SAR   | text-recognition |      Y      |    N     |   N   |   N   |    N     | [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/sar)  |


### Reminder

Note that ncnn, pplnn, and OpenVINO only support the configs of DBNet18 for DBNet.

### FAQs

None
