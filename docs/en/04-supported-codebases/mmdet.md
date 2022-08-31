# MMDetection Support

MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## MMDetection installation tutorial

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation.

## List of MMDetection models supported by MMDeploy

|       Model        |         Task         | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVINO |                                     Model config                                     |
| :----------------: | :------------------: | :---------: | :------: | :--: | :---: | :------: | :----------------------------------------------------------------------------------: |
|        ATSS        |   ObjectDetection    |      Y      |    Y     |  N   |   N   |    Y     |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/atss)     |
|        FCOS        |   ObjectDetection    |      Y      |    Y     |  Y   |   N   |    Y     |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos)     |
|      FoveaBox      |   ObjectDetection    |      Y      |    N     |  N   |   N   |    Y     |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox)   |
|        FSAF        |   ObjectDetection    |      Y      |    Y     |  Y   |   Y   |    Y     |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf)     |
|     RetinaNet      |   ObjectDetection    |      Y      |    Y     |  Y   |   Y   |    Y     |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet)   |
|        SSD         |   ObjectDetection    |      Y      |    Y     |  Y   |   N   |    Y     |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd)      |
|       VFNet        |   ObjectDetection    |      N      |    N     |  N   |   N   |    Y     |    [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/vfnet)     |
|       YOLOv3       |   ObjectDetection    |      Y      |    Y     |  Y   |   N   |    Y     |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo)     |
|       YOLOX        |   ObjectDetection    |      Y      |    Y     |  Y   |   N   |    Y     |    [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox)     |
|   Cascade R-CNN    |   ObjectDetection    |      Y      |    Y     |  N   |   Y   |    Y     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn) |
|    Faster R-CNN    |   ObjectDetection    |      Y      |    Y     |  Y   |   Y   |    Y     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn)  |
| Faster R-CNN + DCN |   ObjectDetection    |      Y      |    Y     |  Y   |   Y   |    Y     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn)  |
|        GFL         |   ObjectDetection    |      Y      |    Y     |  N   |   ?   |    Y     |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl)      |
|     RepPoints      |   ObjectDetection    |      N      |    Y     |  N   |   ?   |    Y     |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints)   |
|        DETR        |   ObjectDetection    |      Y      |    Y     |  N   |   ?   |    Y     |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detr)     |
| Cascade Mask R-CNN | InstanceSegmentation |      Y      |    N     |  N   |   N   |    Y     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn) |
|     Mask R-CNN     | InstanceSegmentation |      Y      |    Y     |  N   |   N   |    Y     |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn)   |
|  Swin Transformer  | InstanceSegmentation |      Y      |    Y     |  N   |   N   |    N     |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/swin)     |
