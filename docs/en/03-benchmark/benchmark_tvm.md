# Test on TVM

## Platform

- Ubuntu 20.04
- tvm 0.9.0

|                                                                         mmcls                                                                          | metric | PyTorch |  TVM  |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :----: | :-----: | :---: |
|                   [ResNet-18](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet/resnet18_b32x8_imagenet.py)                    | top-1  |  69.90  | 69.90 |
|               [ResNeXt-50](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext/resnext50_32x4d_b32x8_imagenet.py)               | top-1  |  77.90  | 77.90 |
| [ShuffleNet V2](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py) | top-1  |  69.55  | 69.55 |
|               [MobileNet V2](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py)               | top-1  |  71.86  | 71.86 |
|     [Vision Transformer](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-base-p16_ft-64xb64_in1k-384.py)     | top-1  |  85.43  | 84.01 |

|                                        mmdet(\*)                                        | metric | PyTorch | TVM  |
| :-------------------------------------------------------------------------------------: | :----: | :-----: | :--: |
| [SSD](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd300_coco.py) | box AP |  25.5   | 25.5 |

\*: We only test model on ssd since dynamic shape is not supported for now.
