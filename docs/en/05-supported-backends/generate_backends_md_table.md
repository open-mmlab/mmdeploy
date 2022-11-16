# How to generate supported-backends markdown table

This tutorial describes how to generate supported-backends markdown table.

## 1.python Environment

```
pip install easydict
```

## 2.Usage

```
python ./tests/regression/generate_md_table.py "${yml_file}" "${output}"
```

### Description

```
yml_file:  input yml config path
output: output markdown file path
```

### Example

Generate backends markdown table from mmseg.yml

```
python tests/regression/generate_md_table.py tests/regression/mmseg.yml tests/regression/mmseg.md
```

## 3.Generated Table

This is an example of mmseg generate backends markdown table

| model                                                                                        | task         | onnxruntime | tensorrt | torchscript | pplnn | openvino | ncnn |
| :------------------------------------------------------------------------------------------- | :----------- | :---------- | :------- | :---------- | :---- | :------- | :--- |
| [FCN](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn)                  | Segmentation | Y           | Y        | Y           | Y     | Y        | Y    |
| [PSPNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet)            | Segmentation | Y           | Y        | Y           | Y     | Y        | Y    |
| [deeplabv3](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3)      | Segmentation | Y           | Y        | Y           | Y     | Y        | Y    |
| [deeplabv3+](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus) | Segmentation | Y           | Y        | Y           | Y     | Y        | Y    |
| [Fast-SCNN](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastscnn)       | Segmentation | Y           | Y        | Y           | Y     | Y        | N    |
| [UNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/unet)                | Segmentation | Y           | Y        | Y           | Y     | Y        | Y    |
| [ANN](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ann)                  | Segmentation | Y           | Y        | Y           | N     | N        | N    |
| [APCNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/apcnet)            | Segmentation | Y           | Y        | Y           | N     | N        | Y    |
| [BiSeNetV1](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/bisenetv1)      | Segmentation | Y           | Y        | Y           | N     | Y        | Y    |
| [BiSeNetV2](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/bisenetv2)      | Segmentation | Y           | Y        | Y           | N     | Y        | Y    |
| [CGNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/cgnet)              | Segmentation | Y           | Y        | Y           | N     | Y        | Y    |
| [EMANet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/emanet)            | Segmentation | Y           | Y        | Y           | N     | Y        | N    |
| [EncNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/encnet)            | Segmentation | Y           | Y        | Y           | N     | Y        | N    |
| [ERFNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/erfnet)            | Segmentation | Y           | Y        | Y           | N     | Y        | Y    |
| [FastFCN](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastfcn)          | Segmentation | Y           | Y        | Y           | N     | Y        | Y    |
| [GCNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/gcnet)              | Segmentation | Y           | Y        | Y           | N     | N        | N    |
| [ICNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/icnet)              | Segmentation | Y           | Y        | Y           | N     | Y        | N    |
| [ISANet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/isanet)            | Segmentation | Y           | Y        | N           | N     | Y        | N    |
| [OCRNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ocrnet)            | Segmentation | Y           | Y        | Y           | N     | Y        | Y    |
| [PointRend](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/point_rend)     | Segmentation | Y           | Y        | Y           | N     | N        | N    |
| [Semantic FPN](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/sem_fpn)     | Segmentation | Y           | Y        | Y           | N     | Y        | Y    |
| [STDC](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/stdc)                | Segmentation | Y           | Y        | Y           | N     | Y        | Y    |
| [UPerNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/upernet)          | Segmentation | Y           | Y        | Y           | N     | N        | N    |
| [Segmenter](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segmenter)      | Segmentation | Y           | Y        | Y           | N     | Y        | Y    |
