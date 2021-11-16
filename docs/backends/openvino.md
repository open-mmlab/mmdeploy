## OpenVINO Support

This tutorial is based on Linux systems like Ubuntu-18.04.

### Installation

1. Install [OpenVINO](https://docs.openvinotoolkit.org/latest/installation_guides.html).
2. Install MMDeploy following the [instructions](../build.md).

### Usage

Example:
```bash
python tools/deploy.py \
    configs/mmdet/single-stage/single-stage_openvino_dynamic.py \
    /mmdetection_dir/mmdetection/configs/ssd/ssd300_coco.py \
    /tmp/snapshots/ssd300_coco_20210803_015428-d231a06e.pth \
    tests/data/tiger.jpeg \
    --work-dir ../deploy_result \
    --device cpu \
    --log-level INFO \
```

### List of supported models exportable to OpenVINO from MMDetection

The table below lists the models that are guaranteed to be exportable to OpenVINO from MMDetection.
|    Model name      |                                  Config                                   | Dynamic Shape |
| :----------------: | :-----------------------------------------------------------------------: | :-----------: |
| ATSS               |                  `configs/atss/atss_r50_fpn_1x_coco.py`                   |       Y       |
| Cascade Mask R-CNN |        `configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py`        |       Y       |
| Cascade R-CNN      |          `configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py`           |       Y       |
| Faster R-CNN       |           `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`            |       Y       |
| FCOS               | `configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco.py` |       Y       |
| FoveaBox           |           `configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py `                |       Y       |
| FSAF               |                  `configs/fsaf/fsaf_r50_fpn_1x_coco.py`                   |       Y       |
| Mask R-CNN         |           `configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py`                |       Y       |
| RetinaNet          |             `configs/retinanet/retinanet_r50_fpn_1x_coco.py`              |       Y       |
| SSD                |                       `configs/ssd/ssd300_coco.py`                        |       Y       |
| YOLOv3             |            `configs/yolo/yolov3_d53_mstrain-608_273e_coco.py`             |       Y       |
| YOLOX              |               `configs/yolox/yolox_tiny_8x8_300e_coco.py`                 |       Y       |
| Faster R-CNN + DCN |            `configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py`       |       Y       |
| VFNet              |                  `configs/vfnet/vfnet_r50_fpn_1x_coco.py`                 |       Y       |

Notes:
- For faster work in OpenVINO in the Faster-RCNN, Mask-RCNN, Cascade-RCNN, Cascade-Mask-RCNN models
the RoiAlign operation is replaced with the [ExperimentalDetectronROIFeatureExtractor](https://docs.openvinotoolkit.org/latest/openvino_docs_ops_detection_ExperimentalDetectronROIFeatureExtractor_6.html) operation in the ONNX graph.
- Models "VFNet" and "Faster R-CNN + DCN" use the custom "DeformableConv2D" operation.
