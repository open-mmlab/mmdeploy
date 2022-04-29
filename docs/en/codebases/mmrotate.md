# MMRotate Support

[MMRotate](https://github.com/open-mmlab/mmrotate) is an open-source toolbox for rotated object detection based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## MMRotate installation tutorial

Please refer to [official installation guide](https://mmrotate.readthedocs.io/en/latest/install.html) to install the codebase.

## MMRotate models support

| Model     | Task          | ONNX Runtime | TensorRT | NCNN | PPLNN | OpenVINO |                                        Model config                                         |
|:----------------------|:--------------|:------------:|:--------:|:----:|:-----:|:--------:|:-------------------------------------------------------------------------------------------:|
| RotatedRetinaNet  | RotatedDetection |      Y       |    N     |  N   |   N   |    N     |   [config](https://github.com/open-mmlab/mmrotate/blob/main/configs/rotated_retinanet/README.md)   |

### Example

```bash
python tools/deploy.py \
configs/mmrotate/rotated-detection_onnxruntime_dynamic.py \
$MMROTATE_DIR/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135.py \
$MMROTATE_DIR/checkpoints/rotated_retinanet_obb_r50_fpn_1x_dota_le135-e4131166.pth \
$MMROTATE_DIR/demo/demo.jpg \
--work-dir work-dirs/mmrotate/rotated_retinanet/ort \
--device cpu
```

Note

- Usually, mmrotate models need some extra information for the input image, but we can't get it directly. So, when exporting the model, you can use `$MMROTATE_DIR/demo/demo.jpg` as input.

## Reminder

None

## FAQs

None
