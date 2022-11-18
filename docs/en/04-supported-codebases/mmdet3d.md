# MMDetection3d Support

MMDetection3d is a next-generation platform for general 3D object detection. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## MMDetection3d installation tutorial

Please refer to [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md) for installation.

## Example

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

## List of MMDetection3d models supported by MMDeploy

|    Model     |      Task      | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVINO |                                      Model config                                      |
| :----------: | :------------: | :---------: | :------: | :--: | :---: | :------: | :------------------------------------------------------------------------------------: |
| PointPillars | VoxelDetection |      Y      |   Y\*    |  N   |   N   |    Y     | [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars) |

## Reminder

1. Voxel detection onnx model excludes model.voxelize layer and model post process, and you can use python api to call these func.

Example:

```python
from mmdeploy.codebase.mmdet3d.deploy import VoxelDetectionModel
VoxelDetectionModel.voxelize(...)
VoxelDetectionModel.post_process(...)
```

2. TensorRT 实测 cu102+TRT8.4 可以正常可视化，cuda11 或 TRT8.2 版本需关注这些 issue。

- [TRT8.2 assertion `is_tensor`](https://github.com/NVIDIA/TensorRT/issues/1541)
- [TRT8.4 output NaN](https://github.com/NVIDIA/TensorRT/issues/2338)
