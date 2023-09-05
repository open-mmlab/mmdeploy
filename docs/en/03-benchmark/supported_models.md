## Supported models

The table below lists the models that are guaranteed to be exportable to other backends.

| Model config                                                                                             | Codebase         | TorchScript | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVINO | Ascend | RKNN |
| :------------------------------------------------------------------------------------------------------- | :--------------- | :---------: | :---------: | :------: | :--: | :---: | :------: | :----: | :--: |
| [RetinaNet](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/retinanet)                        | MMDetection      |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [Faster R-CNN](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/faster_rcnn)                   | MMDetection      |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  N   |
| [YOLOv3](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/yolo)                                | MMDetection      |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   Y    |  Y   |
| [YOLOX](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/yolox)                                | MMDetection      |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [FCOS](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/fcos)                                  | MMDetection      |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [FSAF](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/fsaf)                                  | MMDetection      |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  Y   |
| [Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/mask_rcnn)                       | MMDetection      |      Y      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [SSD](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/ssd)[\*](#note)                         | MMDetection      |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [FoveaBox](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/foveabox)                          | MMDetection      |      Y      |      Y      |    N     |  N   |   N   |    Y     |   N    |  N   |
| [ATSS](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/atss)                                  | MMDetection      |      N      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [GFL](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/gfl)                                    | MMDetection      |      N      |      Y      |    Y     |  N   |   ?   |    Y     |   N    |  N   |
| [Cascade R-CNN](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/cascade_rcnn)                 | MMDetection      |      N      |      Y      |    Y     |  N   |   Y   |    Y     |   N    |  N   |
| [Cascade Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/cascade_rcnn)            | MMDetection      |      N      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [Swin Transformer](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/swin)[\*](#note)           | MMDetection      |      N      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [VFNet](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/vfnet)                                | MMDetection      |      N      |      N      |    N     |  N   |   N   |    Y     |   N    |  N   |
| [RepPoints](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/reppoints)                        | MMDetection      |      N      |      N      |    Y     |  N   |   ?   |    Y     |   N    |  N   |
| [DETR](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/detr)                                  | MMDetection      |      N      |      Y      |    Y     |  N   |   ?   |    N     |   N    |  N   |
| [CenterNet](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/centernet)                        | MMDetection      |      N      |      Y      |    Y     |  N   |   ?   |    Y     |   N    |  N   |
| [SOLO](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/solo)                                  | MMDetection      |      N      |      Y      |    N     |  N   |   N   |    Y     |   N    |  N   |
| [SOLOv2](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/solov2)                              | MMDetection      |      N      |      Y      |    N     |  N   |   N   |    Y     |   N    |  N   |
| [ResNet](https://github.com/open-mmlab/mmpretrain/tree/main/configs/resnet)                              | MMPretrain       |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [ResNeXt](https://github.com/open-mmlab/mmpretrain/tree/main/configs/resnext)                            | MMPretrain       |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [SE-ResNet](https://github.com/open-mmlab/mmpretrain/tree/main/configs/seresnet)                         | MMPretrain       |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [MobileNetV2](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mobilenet_v2)                   | MMPretrain       |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [MobileNetV3](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mobilenet_v3)                   | MMPretrain       |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [ShuffleNetV1](https://github.com/open-mmlab/mmpretrain/tree/main/configs/shufflenet_v1)                 | MMPretrain       |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [ShuffleNetV2](https://github.com/open-mmlab/mmpretrain/tree/main/configs/shufflenet_v2)                 | MMPretrain       |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [VisionTransformer](https://github.com/open-mmlab/mmpretrain/tree/main/configs/vision_transformer)       | MMPretrain       |      Y      |      Y      |    Y     |  Y   |   ?   |    Y     |   Y    |  N   |
| [SwinTransformer](https://github.com/open-mmlab/mmpretrain/tree/main/configs/swin_transformer)           | MMPretrain       |      Y      |      Y      |    Y     |  N   |   ?   |    N     |   ?    |  N   |
| [MobileOne](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mobileone)                        | MMPretrain       |      N      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [FCN](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/fcn)                                | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [PSPNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/pspnet)[\*static](#note)         | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [DeepLabV3](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/deeplabv3)                    | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  N   |
| [DeepLabV3+](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/deeplabv3plus)               | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  N   |
| [Fast-SCNN](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/fastscnn)[\*static](#note)    | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   Y   |    Y     |   N    |  Y   |
| [UNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet)                              | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  Y   |
| [ANN](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/ann)[\*](#note)                     | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [APCNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/apcnet)                          | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    N     |   N    |  Y   |
| [BiSeNetV1](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/bisenetv1)                    | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [BiSeNetV2](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/bisenetv2)                    | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [CGNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/cgnet)                            | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [DMNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/dmnet)                            | MMSegmentation   |      ?      |      Y      |    N     |  N   |   N   |    N     |   N    |  N   |
| [DNLNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/dnlnet)                          | MMSegmentation   |      ?      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [EMANet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/emanet)                          | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [EncNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/encnet)                          | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [ERFNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/erfnet)                          | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [FastFCN](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/fastfcn)                        | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [GCNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/gcnet)                            | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [ICNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/icnet)[\*](#note)                 | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [ISANet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/isanet)[\*static](#note)         | MMSegmentation   |      N      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  Y   |
| [NonLocal Net](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/nonlocal_net)              | MMSegmentation   |      ?      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [OCRNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/ocrnet)                          | MMSegmentation   |      ?      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [PointRend](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/point_rend)                   | MMSegmentation   |      Y      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [Semantic FPN](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/sem_fpn)                   | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [STDC](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/stdc)                              | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  Y   |
| [UPerNet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/upernet)[\*](#note)             | MMSegmentation   |      ?      |      Y      |    Y     |  N   |   N   |    N     |   N    |  Y   |
| [DANet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/danet)                            | MMSegmentation   |      ?      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [Segmenter](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segmenter) [\*static](#note)  | MMSegmentation   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [SRCNN](https://github.com/open-mmlab/mmagic/tree/main/configs/srcnn)                                    | MMagic           |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [ESRGAN](https://github.com/open-mmlab/mmagic/tree/main/configs/esrgan)                                  | MMagic           |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [SRGAN](https://github.com/open-mmlab/mmagic/tree/main/configs/srgan_resnet)                             | MMagic           |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [SRResNet](https://github.com/open-mmlab/mmagic/tree/main/configs/srgan_resnet)                          | MMagic           |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [Real-ESRGAN](https://github.com/open-mmlab/mmagic/tree/main/configs/real_esrgan)                        | MMagic           |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [EDSR](https://github.com/open-mmlab/mmagic/tree/main/configs/edsr)                                      | MMagic           |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [RDN](https://github.com/open-mmlab/mmagic/tree/main/configs/rdn)                                        | MMagic           |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   N    |  N   |
| [DBNet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet)                             | MMOCR            |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |   Y    |  N   |
| [DBNetpp](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnetpp)                         | MMOCR            |      Y      |      Y      |    Y     |  ?   |   ?   |    Y     |   ?    |  N   |
| [PANet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet)                             | MMOCR            |      Y      |      Y      |    Y     |  Y   |   ?   |    Y     |   Y    |  N   |
| [PSENet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet)                           | MMOCR            |      Y      |      Y      |    Y     |  Y   |   ?   |    Y     |   Y    |  N   |
| [TextSnake](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/textsnake)                     | MMOCR            |      Y      |      Y      |    Y     |  Y   |   ?   |    ?     |   ?    |  N   |
| [MaskRCNN](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/maskrcnn)                       | MMOCR            |      Y      |      Y      |    Y     |  ?   |   ?   |    ?     |   ?    |  N   |
| [CRNN](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn)                             | MMOCR            |      Y      |      Y      |    Y     |  Y   |   Y   |    N     |   N    |  N   |
| [SAR](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/sar)                               | MMOCR            |      N      |      Y      |    N     |  N   |   N   |    N     |   N    |  N   |
| [SATRN](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/satrn)                           | MMOCR            |      Y      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [ABINet](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/abinet)                         | MMOCR            |      Y      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [HRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#hrnet-cvpr-2019)         | MMPose           |      N      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [MSPN](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#mspn-arxiv-2019)          | MMPose           |      N      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [LiteHRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#litehrnet-cvpr-2021) | MMPose           |      N      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [Hourglass](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#hourglass-eccv-2016) | MMPose           |      N      |      Y      |    Y     |  Y   |   N   |    Y     |   N    |  N   |
| [SimCC](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#simcc-eccv-2022)        | MMPose           |      N      |      Y      |    Y     |  Y   |   N   |    N     |   N    |  N   |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars)               | MMDetection3d    |      ?      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [CenterPoint (pillar)](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/centerpoint)        | MMDetection3d    |      ?      |      Y      |    Y     |  N   |   N   |    Y     |   N    |  N   |
| [RotatedRetinaNet](https://github.com/open-mmlab/mmrotate/blob/main/configs/rotated_retinanet/README.md) | RotatedDetection |      N      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [Oriented RCNN](https://github.com/open-mmlab/mmrotate/blob/main/configs/oriented_rcnn/README.md)        | RotatedDetection |      N      |      Y      |    Y     |  N   |   N   |    N     |   N    |  N   |
| [Gliding Vertex](https://github.com/open-mmlab/mmrotate/blob/main/configs/gliding_vertex/README.md)      | RotatedDetection |      N      |      N      |    Y     |  N   |   N   |    N     |   N    |  N   |

### Note

- Tag:
  - static: This model only support static export. Please use `static` deploy config, just like $MMDEPLOY_DIR/configs/mmseg/segmentation_tensorrt_static-1024x2048.py.
- SSD: When you convert SSD model, you need to use min shape deploy config just like 300x300-512x512 rather than 320x320-1344x1344, for example $MMDEPLOY_DIR/configs/mmdet/detection/detection_tensorrt_dynamic-300x300-512x512.py.
- YOLOX: YOLOX with ncnn only supports static shape.
- Swin Transformer: For TensorRT, only version 8.4+ is supported.
- SAR: Chinese text recognition model is not supported as the protobuf size of ONNX is limited.
