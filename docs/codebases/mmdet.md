## MMDetection Support

MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMDetection installation tutorial

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation.

### List of MMDetection models supported by MMDeploy

| model        | task         | OnnxRuntime | TensorRT | NCNN | PPL | OpenVINO | model config file(example)                                        |
|:-------------|:-------------|:-----------:|:--------:|:----:|:---:|:--------:|:------------------------------------------------------------------|
| RetinaNet    | single-stage |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/retinanet/retinanet_r50_fpn_1x_coco.py         |
| YOLOv3       | single-stage |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py       |
| FCOS         | single-stage |      Y      |    Y     |  Y   |  N  |    Y     | $MMDET_DIR/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py |
| FSAF         | single-stage |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/fsaf/fsaf_r50_fpn_1x_coco.py                   |
| Faster R-CNN | two-stage    |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py     |
| Mask R-CNN   | two-stage    |      Y      |    Y     |  N   |  Y  |    Y     | $MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py         |

### Reminder

None

### FAQs

None
