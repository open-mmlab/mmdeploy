## OpenVINO Support

This tutorial is based on Linux systems like Ubuntu-18.04.

### Installation

1. Install [OpenVINO](https://docs.openvinotoolkit.org/latest/installation_guides.html).
2. Install MMdeploy following the [instructions](../build.md).

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

|    Model     |                               Config                                | Dynamic Shape |
| :----------: | :-----------------------------------------------------------------: | :-----------: |
|     FCOS     |      `configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py`       |       Y       |
|     FSAF     |               `configs/fsaf/fsaf_r50_fpn_1x_coco.py`                |       Y       |
|  RetinaNet   |          `configs/retinanet/retinanet_r50_fpn_1x_coco.py`           |       Y       |
|     SSD      |                    `configs/ssd/ssd300_coco.py`                     |       Y       |
| Faster R-CNN |        `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`         |       Y       |
| Cascade R-CNN|       `configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py`        |       Y       |

Notes:
- For faster work in OpenVINO in the Faster-RCNN, Cascade-RCNN
the RoiAlign operation is replaced with the [ExperimentalDetectronROIFeatureExtractor](https://docs.openvinotoolkit.org/latest/openvino_docs_ops_detection_ExperimentalDetectronROIFeatureExtractor_6.html) operation in the ONNX graph.
