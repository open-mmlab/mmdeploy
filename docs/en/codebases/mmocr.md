## MMOCR Support

MMOCR is an open-source toolbox based on PyTorch and mmdetection for text detection, text recognition, and the corresponding downstream tasks including key information extraction. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMOCR installation tutorial

Please refer to [install.md](https://mmocr.readthedocs.io/en/latest/install.html) for installation.

### List of MMOCR models supported by MMDeploy

| model |       task       | OnnxRuntime | TensorRT | NCNN | PPLNN | OpenVINO | model config file(example)                                                |
|-------|:----------------:|:-----------:|:--------:|:----:|:---:|:--------:|---------------------------------------------------------------------------|
| DBNet |  text-detection  |      Y      |     Y    |   Y  |  Y  |     Y    | $PATH_TO_MMOCR/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py    |
| CRNN  | text-recognition |      Y      |     Y    |   Y  |  Y  |     N    | $PATH_TO_MMOCR/configs/textrecog/crnn/crnn_academic_dataset.py            |
| SAR   | text-recognition |      Y      |     N    |   N  |  N  |     N    | $PATH_TO_MMOCR/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py |


### Reminder

Note that ncnn, pplnn, and OpenVINO only support the configs of DBNet18 for DBNet.

### FAQs

None
