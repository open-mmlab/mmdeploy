# 模型支持列表

自测完成的 model-backend 组合：

| Model config                                                                                            | Codebase         | TorchScript | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVINO | Ascend | RKNN |
| :------------------------------------------------------------------------------------------------------ | :--------------- | :---------: | :---------: | :------: | :--: | :---: | :------: | :----: | :--: |
| [RetinaNet](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/retinanet)                       | MMDetection      |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [Faster R-CNN](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/faster_rcnn)                  | MMDetection      |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  N   |
| [YOLOv3](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/yolo)                               | MMDetection      |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   Y    |  Y   |
| [YOLOX](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/yolox)                               | MMDetection      |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [FCOS](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/fcos)                                 | MMDetection      |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [FSAF](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/fsaf)                                 | MMDetection      |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  Y   |
| [Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/mask_rcnn)                      | MMDetection      |      Y      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [SSD](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/ssd)[\*](#note)                        | MMDetection      |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [FoveaBox](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/foveabox)                         | MMDetection      |      Y      |      Y      |    N     |  N   |   N   |    Y     |   N    |  N   |
| [ATSS](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/atss)                                 | MMDetection      |      N      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [GFL](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/gfl)                                   | MMDetection      |      N      |      Y      |    Y     |  N   |   ?   |    Y     |   N    |  N   |
| [Cascade R-CNN](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/cascade_rcnn)                | MMDetection      |      N      |      Y      |    Y     |  N   |   Y   |    Y     |   N    |  N   |
| [Cascade Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/cascade_rcnn)           | MMDetection      |      N      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [Swin Transformer](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/swin)[\*](#note)          | MMDetection      |      N      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [VFNet](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/vfnet)                               | MMDetection      |      N      |      N      |    N     |  N   |   N   |    Y     |   N    |  N   |
| [RepPoints](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/reppoints)                       | MMDetection      |      N      |      N      |    Y     |  N   |   ?   |    Y     |   N    |  N   |
| [DETR](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/detr)                                 | MMDetection      |      N      |      Y      |    Y     |  N   |   ?   |    N     |   N    |  N   |
| [CenterNet](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/centernet)                       | MMDetection      |      N      |      Y      |    Y     |  N   |   ?   |    N     |   N    |  N   |
| [SOLO](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/solo)                                 | MMDetection      |      N      |      N      |    N     |  N   |   N   |    Y     |   N    |  N   |
| [ResNet](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/resnet)                        | MMClassification |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [ResNeXt](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/resnext)                      | MMClassification |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [SE-ResNet](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/seresnet)                   | MMClassification |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [MobileNetV2](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/mobilenet_v2)             | MMClassification |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [ShuffleNetV1](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/shufflenet_v1)           | MMClassification |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [ShuffleNetV2](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/shufflenet_v2)           | MMClassification |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [VisionTransformer](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/vision_transformer) | MMClassification |      Y      |      Y      |    Y     |  Y   |   ?   |    Y     |   Y    |  N   |
| [SwinTransformer](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/swin_transformer)     | MMClassification |      Y      |      Y      |    Y     |  N   |   ?   |    N     |   ?    |  N   |
| [MobileOne](https://github.com/open-mmlab/mmclassification/tree/1.x/configs/mobileone)                  | MMClassification |      N      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [FCN](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/fcn)                                | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [PSPNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/pspnet)[\*static](#note)         | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [DeepLabV3](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/deeplabv3)                    | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  N   |
| [DeepLabV3+](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/deeplabv3plus)               | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  N   |
| [Fast-SCNN](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/fastscnn)[\*static](#note)    | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   Y   |    Y     |   N    |  Y   |
| [UNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/unet)                              | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [ANN](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/ann)[\*](#note)                     | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [APCNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/apcnet)                          | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    N     |   N    |  Y   |
| [BiSeNetV1](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/bisenetv1)                    | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [BiSeNetV2](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/bisenetv2)                    | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [CGNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/cgnet)                            | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [DMNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/dmnet)                            | MMSegmentation   |      ?      |      Y      |    N     |  N   |   N   |    N     |   N    |  N   |
| [DNLNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/dnlnet)                          | MMSegmentation   |      ?      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [EMANet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/emanet)                          | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [EncNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/encnet)                          | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [ERFNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/erfnet)                          | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [FastFCN](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/fastfcn)                        | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [GCNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/gcnet)                            | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [ICNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/icnet)[\*](#note)                 | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [ISANet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/isanet)[\*static](#note)         | MMSegmentation   |      N      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  Y   |
| [NonLocal Net](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/nonlocal_net)              | MMSegmentation   |      ?      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [OCRNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/ocrnet)                          | MMSegmentation   |      ?      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [PointRend](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/point_rend)                   | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [Semantic FPN](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/sem_fpn)                   | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [STDC](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/stdc)                              | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [UPerNet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/upernet)[\*](#note)             | MMSegmentation   |      ?      |      Y      |    Y     |  N   |   N   |    N     |   N    |  Y   |
| [DANet](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/danet)                            | MMSegmentation   |      ?      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [Segmenter](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/segmenter) [\*static](#note)  | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [SRCNN](https://github.com/open-mmlab/mmediting/tree/1.x/configs/srcnn)                                 | MMEditing        |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [ESRGAN](https://github.com/open-mmlab/mmediting/tree/1.x/configs/esrgan)                               | MMEditing        |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [SRGAN](https://github.com/open-mmlab/mmediting/tree/1.x/configs/srgan_resnet)                          | MMEditing        |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [SRResNet](https://github.com/open-mmlab/mmediting/tree/1.x/configs/srgan_resnet)                       | MMEditing        |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [Real-ESRGAN](https://github.com/open-mmlab/mmediting/tree/1.x/configs/real_esrgan)                     | MMEditing        |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [EDSR](https://github.com/open-mmlab/mmediting/tree/1.x/configs/edsr)                                   | MMEditing        |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [RDN](https://github.com/open-mmlab/mmediting/tree/1.x/configs/rdn)                                     | MMEditing        |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [DBNet](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/dbnet)                             | MMOCR            |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  N   |
| [PANet](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/panet)                             | MMOCR            |      Y      |      Y      |    Y     |  Y   |   ?   |    Y     |   Y    |  N   |
| [PSENet](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textdet/psenet)                           | MMOCR            |      Y      |      Y      |    Y     |  Y   |   ?   |    Y     |   Y    |  N   |
| [CRNN](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textrecog/crnn)                             | MMOCR            |      Y      |      Y      |    Y     |  Y   |   Y   |    N     |   N    |  N   |
| [SAR](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textrecog/sar)                               | MMOCR            |      N      |      Y      |    N     |  N   |   N   |    N     |   N    |  N   |
| [SATRN](https://github.com/open-mmlab/mmocr/blob/1.x/configs/textrecog/satrn)                           | MMOCR            |      Y      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [HRNet](https://mmpose.readthedocs.io/en/1.x/model_zoo_papers/backbones.html#hrnet-cvpr-2019)           | MMPose           |      N      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [MSPN](https://mmpose.readthedocs.io/en/1.x/model_zoo_papers/backbones.html#mspn-arxiv-2019)            | MMPose           |      N      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [LiteHRNet](https://mmpose.readthedocs.io/en/1.x/model_zoo_papers/backbones.html#litehrnet-cvpr-2021)   | MMPose           |      N      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [Hourglass](https://mmpose.readthedocs.io/en/1.x/model_zoo_papers/backbones.html#hourglass-eccv-2016)   | MMPose           |      N      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [SimCC](https://mmpose.readthedocs.io/en/1.x/model_zoo_papers/algorithms.html#simcc-eccv-2022)          | MMPose           |      N      |      Y      |    Y     |  Y   |   N   |    N     |   N    |  N   |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/dev-1.x/configs/pointpillars)           | MMDetection3d    |      ?      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [CenterPoint (pillar)](https://github.com/open-mmlab/mmdetection3d/tree/dev-1.x/configs/centerpoint)    | MMDetection3d    |      ?      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [RotatedRetinaNet](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/rotated_retinanet/README.md) | RotatedDetection |      N      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [Oriented RCNN](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/oriented_rcnn/README.md)        | RotatedDetection |      N      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [Gliding Vertex](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/gliding_vertex/README.md)      | RotatedDetection |      N      |      N      |    Y     |  N   |   N   |    N     |   N    |  N   |

## Note

- Tag:
  - static: This model only support static export. Please use `static` deploy config, just like $MMDEPLOY_DIR/configs/mmseg/segmentation_tensorrt_static-1024x2048.py.
- SSD: When you convert SSD model, you need to use min shape deploy config just like 300x300-512x512 rather than 320x320-1344x1344, for example $MMDEPLOY_DIR/configs/mmdet/detection/detection_tensorrt_dynamic-300x300-512x512.py.
- YOLOX: YOLOX with ncnn only supports static shape.
- Swin Transformer: For TensorRT, only version 8.4+ is supported.
- SAR: Chinese text recognition model is not supported as the protobuf size of ONNX is limited.
