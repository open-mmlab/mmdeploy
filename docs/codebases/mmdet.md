## MMDetection Support

MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMDetection installation tutorial

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation.

### List of MMDetection models supported by MMDeploy

|    model     |       task       | model config file(example)                                                                | OnnxRuntime |    TensorRT   | NCNN |  PPL  |
| :----------: | :--------------: | :---------------------------------------------------------------------------------------: | :---------: | :-----------: | :---:| :---: |
| RetinaNet    | single-stage     | $PATH_TO_MMDET/configs/retinanet/retinanet_r50_fpn_1x_coco.py                             |      Y      |       Y       |   Y  |   Y   |
| Faster R-CNN | two-stage        | $PATH_TO_MMDET/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py                         |      Y      |       Y       |   Y  |   Y   |
| YOLOv3       | single-stage     | $PATH_TO_MMDET/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py                           |      Y      |       Y       |   N  |   Y   |
| FCOS         | single-stage     | $PATH_TO_MMDET/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py                     |      Y      |       Y       |   Y  |   N   |
| FSAF         | single-stage     | $PATH_TO_MMDET/configs/fsaf/fsaf_r50_fpn_1x_coco.py                                       |      Y      |       Y       |   Y  |   Y   |
| Mask R-CNN   | two-stage        | $PATH_TO_MMDET/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py                             |      Y      |       Y       |   N  |   Y   |

### Reminder

None

### FAQs

None
