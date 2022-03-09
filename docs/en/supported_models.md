## List of supported models exportable to other backends

The table below lists the models that are guaranteed to be exportable to other backends.

| Model                     | Codebase         | OnnxRuntime | TensorRT | NCNN | PPLNN | OpenVINO |                                          Model config                                          |
|:--------------------------|:-----------------|:-----------:|:--------:|:----:|:-----:|:--------:|:----------------------------------------------------------------------------------------------:|
| RetinaNet                 | MMDetection      |      Y      |    Y     |  Y   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet)        |
| Faster R-CNN              | MMDetection      |      Y      |    Y     |  Y   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn)       |
| YOLOv3                    | MMDetection      |      Y      |    Y     |  Y   |   N   |    Y     |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo)          |
| YOLOX                     | MMDetection      |      Y      |    Y     |  Y   |   N   |    Y     |         [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox)          |
| FCOS                      | MMDetection      |      Y      |    Y     |  Y   |   N   |    Y     |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos)          |
| FSAF                      | MMDetection      |      Y      |    Y     |  Y   |   Y   |    Y     |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf)          |
| Mask R-CNN                | MMDetection      |      Y      |    Y     |  N   |   N   |    Y     |       [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn)        |
| SSD[*](#note)             | MMDetection      |      Y      |    Y     |  Y   |   N   |    Y     |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd)           |
| FoveaBox                  | MMDetection      |      Y      |    N     |  N   |   N   |    Y     |        [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox)        |
| ATSS                      | MMDetection      |      Y      |    Y     |  N   |   N   |    Y     |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/atss)          |
| GFL                       | MMDetection      |      Y      |    Y     |  N   |   ?   |    Y     |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl)           |
| Cascade R-CNN             | MMDetection      |      Y      |    Y     |  N   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn)      |
| Cascade Mask R-CNN        | MMDetection      |      Y      |    N     |  N   |   N   |    Y     |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn)      |
| VFNet                     | MMDetection      |      N      |    N     |  N   |   N   |    Y     |         [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/vfnet)          |
| ResNet                    | MMClassification |      Y      |    Y     |  Y   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)       |
| ResNeXt                   | MMClassification |      Y      |    Y     |  Y   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)      |
| SE-ResNet                 | MMClassification |      Y      |    Y     |  Y   |   Y   |    Y     |     [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet)      |
| MobileNetV2               | MMClassification |      Y      |    Y     |  Y   |   Y   |    Y     |   [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2)    |
| ShuffleNetV1              | MMClassification |      Y      |    Y     |  Y   |   Y   |    Y     |   [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v1)   |
| ShuffleNetV2              | MMClassification |      Y      |    Y     |  Y   |   Y   |    Y     |   [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/shufflenet_v2)   |
| FCN                       | MMSegmentation   |      Y      |    Y     |  Y   |   Y   |    Y     |         [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn)         |
| PSPNet[*static](#note)    | MMSegmentation   |      Y      |    Y     |  Y   |   Y   |    Y     |       [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet)        |
| DeepLabV3                 | MMSegmentation   |      Y      |    Y     |  Y   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3)      |
| DeepLabV3+                | MMSegmentation   |      Y      |    Y     |  Y   |   Y   |    Y     |    [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus)    |
| Fast-SCNN[*static](#note) | MMSegmentation   |      Y      |    Y     |  N   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fastscnn)       |
| UNet[*static](#note)      | MMSegmentation   |      Y      |    Y     |  Y   |   Y   |    Y     |        [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/unet)         |
| SRCNN                     | MMEditing        |      Y      |    Y     |  Y   |   Y   |    Y     |     [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srcnn)      |
| ESRGAN                    | MMEditing        |      Y      |    Y     |  Y   |   Y   |    Y     |     [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/esrgan)     |
| SRGAN                     | MMEditing        |      Y      |    Y     |  Y   |   Y   |    Y     | [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan) |
| SRResNet                  | MMEditing        |      Y      |    Y     |  Y   |   Y   |    Y     | [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan) |
| Real-ESRGAN               | MMEditing        |      Y      |    Y     |  Y   |   Y   |    Y     |  [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/real_esrgan)   |
| EDSR                      | MMEditing        |      Y      |    Y     |  Y   |   N   |    Y     |      [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/edsr)      |
| RDN                       | MMEditing        |      Y      |    Y     |  Y   |   Y   |    Y     |      [config](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/rdn)       |
| DBNet                     | MMOCR            |      Y      |    Y     |  Y   |   Y   |    Y     |         [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/dbnet)          |
| CRNN                      | MMOCR            |      Y      |    Y     |  Y   |   Y   |    N     |         [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/crnn)         |
| SAR                       | MMOCR            |      Y      |    N     |  N   |   N   |    N     |         [config](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/sar)          |
| HRNet                     | MMPose           |      Y      |    Y     |  Y   |   N   |    Y     |    [config](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrnet-cvpr-2019)     |
| MSPN                      | MMPose           |      Y      |    Y     |  Y   |   N   |    Y     |    [config](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#mspn-arxiv-2019)     |
| LiteHRNet                 | MMPose           |      Y      |    Y     |  N   |   N   |    Y     |  [config](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#litehrnet-cvpr-2021)   |

### Note

- Tag:
  - static: This model only support static export. Please use `static` deploy config, just like $MMDEPLOY_DIR/configs/mmseg/segmentation_tensorrt_static-1024x2048.py.
- SSD: When you convert SSD model, you need to use min shape deploy config just like 300x300-512x512 rather than 320x320-1344x1344, for example $MMDEPLOY_DIR/configs/mmdet/detection/detection_tensorrt_dynamic-300x300-512x512.py.
- YOLOX: YOLOX with ncnn only supports static shape.
