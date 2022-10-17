# mmdet 模型支持列表

mmdet 是基于 pytorch 的检测工具箱，属于 [OpenMMLab](https://openmmlab.com/)。

## 安装 mmdet

请参照 [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) 。

## 支持列表

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
