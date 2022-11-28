# mmpose 模型支持列表

[mmpose](https://github.com/open-mmlab/mmpose) 是一个基于 PyTorch 的姿态估计的开源工具箱，也是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

## 安装 mmpose

参照 [official installation guide](https://mmpose.readthedocs.io/en/latest/install.html)。

## 支持列表

| Model     | Task          | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |                                        Model config                                         |
| :-------- | :------------ | :----------: | :------: | :--: | :---: | :------: | :-----------------------------------------------------------------------------------------: |
| HRNet     | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |   [config](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrnet-cvpr-2019)   |
| MSPN      | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |   [config](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#mspn-arxiv-2019)   |
| LiteHRNet | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     | [config](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#litehrnet-cvpr-2021) |

### 使用方法

```bash
python tools/deploy.py \
configs/mmpose/posedetection_tensorrt_static-256x192.py \
$MMPOSE_DIR/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
$MMPOSE_DIR/checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
$MMDEPLOY_DIR/demo/resources/human-pose.jpg \
--work-dir work-dirs/mmpose/topdown/hrnet/trt \
--device cuda
```

注意事项

- mmpose 模型需要额外的输入，但我们无法直接获取它。在导出模型时，可以使用 `$MMDEPLOY_DIR/demo/resources/human-pose.jpg` 作为输入。
