# TVM 测试

## 支持模型列表

| Model             | Codebase         |                                          Model config                                           |
| :---------------- | :--------------- | :---------------------------------------------------------------------------------------------: |
| RetinaNet         | MMDetection      |        [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet)        |
| Faster R-CNN      | MMDetection      |       [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn)       |
| YOLOv3            | MMDetection      |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo)           |
| YOLOX             | MMDetection      |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox)          |
| Mask R-CNN        | MMDetection      |        [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn)        |
| SSD               | MMDetection      |           [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd)           |
| ResNet            | MMClassification |       [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)       |
| ResNeXt           | MMClassification |      [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)       |
| SE-ResNet         | MMClassification |      [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)      |
| MobileNetV2       | MMClassification |    [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2)    |
| ShuffleNetV1      | MMClassification |   [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v1)    |
| ShuffleNetV2      | MMClassification |   [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2)    |
| VisionTransformer | MMClassification | [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/vision_transformer) |
| FCN               | MMSegmentation   |         [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn)          |
| PSPNet            | MMSegmentation   |        [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet)        |
| DeepLabV3         | MMSegmentation   |      [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3)       |
| DeepLabV3+        | MMSegmentation   |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus)     |
| UNet              | MMSegmentation   |         [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/unet)         |

表中仅列出已测试模型，未列出的模型可能同样支持，可以自行尝试转换。

## Test

- Ubuntu 20.04
- tvm 0.9.0

|                                                                         mmcls                                                                          | metric | PyTorch |  TVM  |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :----: | :-----: | :---: |
|                   [ResNet-18](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet/resnet18_b32x8_imagenet.py)                    | top-1  |  69.90  | 69.90 |
|               [ResNeXt-50](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext/resnext50_32x4d_b32x8_imagenet.py)               | top-1  |  77.90  | 77.90 |
| [ShuffleNet V2](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py) | top-1  |  69.55  | 69.55 |
|               [MobileNet V2](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py)               | top-1  |  71.86  | 71.86 |

<!-- |     [Vision Transformer](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-base-p16_ft-64xb64_in1k-384.py)     | top-1  |  85.43  | 84.01 | -->

|                                        mmdet(\*)                                        | metric | PyTorch | TVM  |
| :-------------------------------------------------------------------------------------: | :----: | :-----: | :--: |
| [SSD](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd300_coco.py) | box AP |  25.5   | 25.5 |

\*: 由于暂时不支持动态转换，因此仅提供 SSD 的精度测试结果。

|                                                           mmseg                                                            | metric | PyTorch |  TVM  |
| :------------------------------------------------------------------------------------------------------------------------: | :----: | :-----: | :---: |
|     [FCN](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py)      |  mIoU  |  72.25  | 72.36 |
| [PSPNet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py) |  mIoU  |  78.55  | 77.90 |
