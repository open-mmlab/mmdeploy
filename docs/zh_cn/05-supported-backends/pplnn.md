# PPLNN 支持情况

MMDeploy supports ppl.nn v0.9.1 and later. This tutorial is based on Linux systems like Ubuntu-18.04.

## Installation

1. Please install [pyppl](https://github.com/openppl-public/ppl.nn) following [install-guide](https://github.com/openppl-public/ppl.nn/blob/master/docs/en/building-from-source.md).

2. Install MMDeploy following the [instructions](../01-how-to-build/build_from_source.md).

## Usage

Example:

```bash
python tools/deploy.py \
    configs/mmdet/detection/detection_pplnn_dynamic-800x1344.py \
    /mmdetection_dir/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    /tmp/snapshots/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth \
    tests/data/tiger.jpeg \
    --work-dir ../deploy_result \
    --device cuda \
    --log-level INFO
```
