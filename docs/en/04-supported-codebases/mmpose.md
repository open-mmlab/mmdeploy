# MMPose Support

[MMPose](https://github.com/open-mmlab/mmpose) is an open-source toolbox for pose estimation based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## MMPose installation tutorial

Please refer to [official installation guide](https://mmpose.readthedocs.io/en/latest/install.html) to install the codebase.

## MMPose models support

| Model     | Task          | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |                                        Model config                                         |
| :-------- | :------------ | :----------: | :------: | :--: | :---: | :------: | :-----------------------------------------------------------------------------------------: |
| HRNet     | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |   [config](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrnet-cvpr-2019)   |
| MSPN      | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |   [config](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#mspn-arxiv-2019)   |
| LiteHRNet | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     | [config](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#litehrnet-cvpr-2021) |

### Example

```bash
python tools/deploy.py \
configs/mmpose/posedetection_tensorrt_static-256x192.py \
$MMPOSE_DIR/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
$MMPOSE_DIR/checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
$MMDEPLOY_DIR/demo/resources/human-pose.jpg \
--work-dir work-dirs/mmpose/topdown/hrnet/trt \
--device cuda
```

Note

- Usually, mmpose models need some extra information for the input image, but we can't get it directly. So, when exporting the model, you can use `$MMDEPLOY_DIR/demo/resources/human-pose.jpg` as input.
