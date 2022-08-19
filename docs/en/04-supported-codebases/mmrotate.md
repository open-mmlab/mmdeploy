# MMRotate Support

[MMRotate](https://github.com/open-mmlab/mmrotate) is an open-source toolbox for rotated object detection based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## MMRotate installation tutorial

Please refer to [official installation guide](https://mmrotate.readthedocs.io/en/latest/install.html) to install the codebase.

## MMRotate models support

| Model            | Task             | ONNX Runtime | TensorRT | NCNN | PPLNN | OpenVINO |                                          Model config                                          |
| :--------------- | :--------------- | :----------: | :------: | :--: | :---: | :------: | :--------------------------------------------------------------------------------------------: |
| RotatedRetinaNet | RotatedDetection |      Y       |    Y     |  N   |   N   |    N     | [config](https://github.com/open-mmlab/mmrotate/blob/main/configs/rotated_retinanet/README.md) |
| Oriented RCNN    | RotatedDetection |      Y       |    Y     |  N   |   N   |    N     |   [config](https://github.com/open-mmlab/mmrotate/blob/main/configs/oriented_rcnn/README.md)   |
| Gliding Vertex   | RotatedDetection |      N       |    Y     |  N   |   N   |    N     |  [config](https://github.com/open-mmlab/mmrotate/blob/main/configs/gliding_vertex/README.md)   |
| RoI Transformer  | RotatedDetection |      Y       |    Y     |  N   |   N   |    N     |     [config](https://github.com/open-mmlab/mmrotate/blob/main/configs/roi_trans/README.md)     |

### Example

```bash
# convert ort
python tools/deploy.py \
configs/mmrotate/rotated-detection_onnxruntime_dynamic.py \
$MMROTATE_DIR/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135.py \
$MMROTATE_DIR/checkpoints/rotated_retinanet_obb_r50_fpn_1x_dota_le135-e4131166.pth \
$MMROTATE_DIR/demo/demo.jpg \
--work-dir work-dirs/mmrotate/rotated_retinanet/ort \
--device cpu

# compute metric
python tools/test.py \
    configs/mmrotate/rotated-detection_onnxruntime_dynamic.py \
    $MMROTATE_DIR/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135.py \
    --model work-dirs/mmrotate/rotated_retinanet/ort/end2end.onnx \
    --metrics mAP

# generate submit file
python tools/test.py \
    configs/mmrotate/rotated-detection_onnxruntime_dynamic.py \
    $MMROTATE_DIR/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135.py \
    --model work-dirs/mmrotate/rotated_retinanet/ort/end2end.onnx \
    --format-only \
    --metric-options submission_dir=work-dirs/mmrotate/rotated_retinanet/ort/Task1_results
```

Note

- Usually, mmrotate models need some extra information for the input image, but we can't get it directly. So, when exporting the model, you can use `$MMROTATE_DIR/demo/demo.jpg` as input.
