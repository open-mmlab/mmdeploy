# MMDetection3d Deployment

- [MMDetection3d Deployment](#mmdetection3d-deployment)
  - [Install mmdet3d](#install-mmdet3d)
  - [Convert model](#convert-model)
  - [Model inference](#model-inference)
  - [Supported models](#supported-models)

______________________________________________________________________

[MMDetection3d](https://github.com/open-mmlab/mmdetection3d) aka `mmdet3d` is an open source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D detection. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## Install mmdet3d

We could install mmdet3d through [mim](https://github.com/open-mmlab/mim).
For other ways of installation, please refer to [here](https://mmdetection3d.readthedocs.io/en/latest/get_started.html#installation)

```bash
python3 -m pip install -U openmim
python3 -m mim install "mmdet3d>=1.1.0"
```

## Convert model

For example, use `tools/deploy.py` to convert centerpoint to onnxruntime format

```bash
# cd to mmdeploy root directory
# download config and model
mim download mmdet3d --config centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d --dest .

export MODEL_CONFIG=centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py

export MODEL_PATH=centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth

export TEST_DATA=tests/data/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151612397179.pcd.bin

python3 tools/deploy.py configs/mmdet3d/voxel-detection/voxel-detection_onnxruntime_dynamic.py $MODEL_CONFIG $MODEL_PATH $TEST_DATA --work-dir centerpoint
```

This step would generate `end2end.onnx` in `work-dir`

```bash
ls -lah centerpoint
..
-rw-rw-r--  1 rg rg  87M 11月  4 19:48 end2end.onnx
```

## Model inference

At present, the voxelize preprocessing and postprocessing of mmdet3d are not converted into onnx operations; the C++ SDK has not yet implemented the voxelize calculation.

The caller needs to refer to the corresponding [python implementation](../../../mmdeploy/codebase/mmdet3d/deploy/voxel_detection_model.py) to complete.

## Supported models

|                                                                                model                                                                                 |        task         | dataset  | onnxruntime | openvino | tensorrt\* |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------: | :------: | :---------: | :------: | :--------: |
| [centerpoint](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py) |   voxel detection   | nuScenes |     ✔️      |    ✔️    |     ✔️     |
|             [pointpillars](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py)              |   voxel detection   | nuScenes |     ✔️      |    ✔️    |     ✔️     |
|            [pointpillars](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py)            |   voxel detection   |  KITTI   |     ✔️      |    ✔️    |     ✔️     |
|                   [smoke](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/smoke/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py)                    | monocular detection |  KITTI   |     ✔️      |    x     |     ✔️     |

- Make sure trt >= 8.6 for some bug fixed, such as ScatterND, dynamic shape crash and so on.
