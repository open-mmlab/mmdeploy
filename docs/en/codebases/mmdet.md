## MMDetection Support

MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMDetection installation tutorial

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation.

### List of MMDetection models supported by MMDeploy

| model              | task         | OnnxRuntime | TensorRT | NCNN | PPLNN | OpenVINO | model config file(example)                                           |
|:-------------------|:-------------|:-----------:|:--------:|:----:|:---:|:--------:|:---------------------------------------------------------------------|
| ATSS               | single-stage |      Y      |    ?     |  ?   |  ?  |    Y     | $MMDET_DIR/configs/atss/atss_r50_fpn_1x_coco.py                      |
| FCOS               | single-stage |      Y      |    Y     |  Y   |  N  |    Y     | $MMDET_DIR/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py    |
| FoveaBox           | single-stage |      Y      |    ?     |  ?   |  ?  |    Y     | $MMDET_DIR/configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py             |
| FSAF               | single-stage |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/fsaf/fsaf_r50_fpn_1x_coco.py                      |
| RetinaNet          | single-stage |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/retinanet/retinanet_r50_fpn_1x_coco.py            |
| SSD                | single-stage |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/ssd/ssd300_coco.py                                |
| VFNet              | single-stage |      Y      |    ?     |  ?   |  ?  |    Y     | $MMDET_DIR/configs/vfnet/vfnet_r50_fpn_1x_coco.py                    |
| YOLOv3             | single-stage |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py          |
| YOLOX              | single-stage |      Y      |    ?     |  ?   |  ?  |    Y     | $MMDET_DIR/configs/yolox/yolox_tiny_8x8_300e_coco.py                 |
| Cascade R-CNN      | two-stage    |      Y      |    ?     |  ?   |  Y  |    Y     | $MMDET_DIR/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py      |
| Faster R-CNN       | two-stage    |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py        |
| Faster R-CNN + DCN | two-stage    |      Y      |    Y     |  Y   |  Y  |    Y     | $MMDET_DIR/configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py`   |
| Mask Cascade R-CNN | two-stage    |      Y      |    ?     |  ?   |  Y  |    Y     | $MMDET_DIR/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py |
| Mask R-CNN         | two-stage    |      Y      |    Y     |  N   |  Y  |    Y     | $MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py            |

### Reminder

None

### FAQs

None
