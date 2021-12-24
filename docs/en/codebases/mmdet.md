## MMDetection Support

MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMDetection installation tutorial

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation.

### List of MMDetection models supported by MMDeploy

| model              |         task         | OnnxRuntime | TensorRT | NCNN | PPLNN | OpenVINO | model config file(example)                                           |
|:------------------:|:--------------------:|:-----------:|:--------:|:----:|:-----:|:--------:|:---------------------------------------------------------------------|
| ATSS               | ObjectDetection      |      Y      |    Y     |  N   |   N   |    Y     | $MMDET_DIR/configs/atss/atss_r50_fpn_1x_coco.py                      |
| FCOS               | ObjectDetection      |      Y      |    Y     |  Y   |   N   |    Y     | $MMDET_DIR/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py    |
| FoveaBox           | ObjectDetection      |      Y      |    N     |  N   |   N   |    Y     | $MMDET_DIR/configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py             |
| FSAF               | ObjectDetection      |      Y      |    Y     |  Y   |   Y   |    Y     | $MMDET_DIR/configs/fsaf/fsaf_r50_fpn_1x_coco.py                      |
| RetinaNet          | ObjectDetection      |      Y      |    Y     |  Y   |   Y   |    Y     | $MMDET_DIR/configs/retinanet/retinanet_r50_fpn_1x_coco.py            |
| SSD                | ObjectDetection      |      Y      |    Y     |  Y   |   N   |    Y     | $MMDET_DIR/configs/ssd/ssd300_coco.py                                |
| VFNet              | ObjectDetection      |      N      |    N     |  N   |   N   |    Y     | $MMDET_DIR/configs/vfnet/vfnet_r50_fpn_1x_coco.py                    |
| YOLOv3             | ObjectDetection      |      Y      |    Y     |  Y   |   N   |    Y     | $MMDET_DIR/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py          |
| YOLOX              | ObjectDetection      |      Y      |    Y     |  N   |   N   |    Y     | $MMDET_DIR/configs/yolox/yolox_tiny_8x8_300e_coco.py                 |
| Cascade R-CNN      | ObjectDetection      |      Y      |    Y     |  N   |   Y   |    Y     | $MMDET_DIR/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py      |
| Faster R-CNN       | ObjectDetection      |      Y      |    Y     |  Y   |   Y   |    Y     | $MMDET_DIR/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py        |
| Faster R-CNN + DCN | ObjectDetection      |      Y      |    Y     |  Y   |   Y   |    Y     | $MMDET_DIR/configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py`   |
| Cascade Mask R-CNN | InstanceSegmentation |      Y      |    N     |  N   |   N   |    Y     | $MMDET_DIR/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py |
| Mask R-CNN         | InstanceSegmentation |      Y      |    Y     |  N   |   N   |    Y     | $MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py            |

### Reminder

None

### FAQs

None
