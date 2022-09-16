# mmdet3d 模型支持列表

MMDetection3d是用于通用 3D 物体检测平台。属于 [OpenMMLab](https://openmmlab.com/)。

## 安装 mmdet3d

参照 [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md)。

## 示例

```bash
export MODEL_PATH=https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth

python tools/deploy.py \
       configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic.py \
       ${MMDET3D_DIR}/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py \
       ${MODEL_PATH} \
       ${MMDET3D_DIR}/demo/data/kitti/kitti_000008.bin \
        --work-dir \
        work_dir \
        --show \
        --device \
        cuda:0
```

## 支持列表

|    Model     |      Task      | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVINO |                                      Model config                                      |
| :----------: | :------------: | :---------: | :------: | :--: | :---: | :------: | :------------------------------------------------------------------------------------: |
| PointPillars | VoxelDetection |      Y      |    Y     |  N   |   N   |    Y     | [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars) |

## 注意事项

体素检测 onnx 模型不包含 model.voxelize 层和模型后处理，可用 python api 来调这些函数。

示例：

```python
from mmdeploy.codebase.mmdet3d.deploy import VoxelDetectionModel
VoxelDetectionModel.voxelize(...)
VoxelDetectionModel.post_process(...)
```
