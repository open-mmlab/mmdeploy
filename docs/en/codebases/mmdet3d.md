## MMDetection3d Support

MMDetection3d is a next-generation platform for general 3D object detection. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

### MMDetection3d installation tutorial

Please refer to [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md) for installation.

### List of MMDetection3d models supported by MMDeploy

|       Model        |         Task         | OnnxRuntime | TensorRT | NCNN  | PPLNN | OpenVINO |                                     Model config                                                         |
| :----------------: | :------------------: | :---------: | :------: | :---: | :---: | :------: | :------------------------------------------------------------------------------------------------------: |
|    PointPillars    |   VoxelDetection     |      Y      |    Y     |   N   |   N   |    Y     |     [config](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/pointpillars)     |

### Reminder

Voxel detection onnx model excludes model.voxelize layer and model post process, and you can use python api to call these func.

Example:

```python
from mmdeploy.codebase.mmdet3d.deploy import VoxelDetectionModel
VoxelDetectionModel.voxelize(...)
VoxelDetectionModel.post_process(...)
```

### FAQs

None
