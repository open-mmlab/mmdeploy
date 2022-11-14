# MMDetection3d Deployment

- [Install mmdet3d](#install-mmdet3d)
- [Convert model](#convert-model)
- [Model inference](#model-inference)
- [Supported models](#supported-models)

______________________________________________________________________

[MMDetection3d](https://github.com/open-mmlab/mmdetection3d) aka `mmdet3d` is an open source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D detection. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## Install mmdet3d

These branches are required for mmdet3d deployment

| codebase |  commit   |
| :------: | :-------: |
| mmdet3d  | v1.1.0rc1 |
|   mmcv   | v2.0.0rc1 |
|  mmdet   | v3.0.0rc1 |
|  mmseg   | v1.0.0rc0 |

First checkout and install mmcv/mmdet/mmseg/mmdet3d

```bash
python3 -m pip install openmim --user
python3 -m mim install mmcv==2.0.0rc1 mmdet==3.0.0rc1 mmseg==1.0.0rc0 --user

git clone https://github.com/open-mmlab/mmdetection3d --branch v1.1.0rc1
cd mmdetection3d
python3 -m pip install .
cd -
```

After installation, `tools/check_env.py` should display mmdet3d version normally

```bash
python3 tools/check_env.py
..
11/11 13:56:19 - mmengine - INFO - **********Codebase information**********
11/11 13:56:19 - mmengine - INFO - mmdet:       3.0.0rc1
11/11 13:56:19 - mmengine - INFO - mmseg:       1.0.0rc0
..
11/11 13:56:19 - mmengine - INFO - mmdet3d:     1.1.0rc1
```

## Convert model

For example, use `tools/deploy.py` to convert centerpoint to onnxruntime format

```bash
export MODEL_CONFIG=/path/to/mmdetection3d/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py

export MODEL_PATH=https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210816_064624-0f3299c0.pth

export TEST_DATA=/path/to/mmdetection3d/tests/data/nuscenes/sweeps/LIDAR_TOP/n008-2018-09-18-12-07-26-0400__LIDAR_TOP__1537287083900561.pcd.bin

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

|                                                                                  model                                                                                  | dataset  | onnxruntime | openvino | tensorrt\* |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------: | :---------: | :------: | :--------: |
| [centerpoint](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py) | nuScenes |     ✔️      |    ✔️    |     ✔️     |
|             [pointpillars](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py)              | nuScenes |     ✔️      |    ✔️    |     ✔️     |
|            [pointpillars](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py)            |  KITTI   |     ✔️      |    ✔️    |     ✔️     |

- Make sure trt >= 8.4 for some bug fixed, such as ScatterND, dynamic shape crash and so on.
