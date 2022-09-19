# mmrotate 模型支持列表

[mmrotate](https://github.com/open-mmlab/mmrotate) 是一个基于 PyTorch 的旋转物体检测的开源工具箱，也是 [OpenMMLab](https://openmmlab.com/)  项目的一部分。

## 安装 mmrotate

参照 [official installation guide](https://mmrotate.readthedocs.io/en/latest/install.html)。

## 支持列表

| Model            | Task             | ONNX Runtime | TensorRT | NCNN | PPLNN | OpenVINO |                                          Model config                                          |
| :--------------- | :--------------- | :----------: | :------: | :--: | :---: | :------: | :--------------------------------------------------------------------------------------------: |
| RotatedRetinaNet | RotatedDetection |      Y       |    Y     |  N   |   N   |    N     | [config](https://github.com/open-mmlab/mmrotate/blob/main/configs/rotated_retinanet/README.md) |
| Oriented RCNN    | RotatedDetection |      Y       |    Y     |  N   |   N   |    N     |   [config](https://github.com/open-mmlab/mmrotate/blob/main/configs/oriented_rcnn/README.md)   |
| Gliding Vertex   | RotatedDetection |      N       |    Y     |  N   |   N   |    N     |  [config](https://github.com/open-mmlab/mmrotate/blob/main/configs/gliding_vertex/README.md)   |
| RoI Transformer  | RotatedDetection |      Y       |    Y     |  N   |   N   |    N     |     [config](https://github.com/open-mmlab/mmrotate/blob/main/configs/roi_trans/README.md)     |

### 使用举例

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

注意：

- mmrotate 模型需要额外输入，但我们无法直接获取它。在导出模型时，可以使用 `$MMROTATE_DIR/demo/demo.jpg` 作为输入。
