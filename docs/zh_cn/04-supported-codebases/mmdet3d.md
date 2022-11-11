# MMDetection3d 模型部署

- [安装 mmdet3d](#安装-mmdet3d)
- [模型转换](#模型转换)
- [模型推理](#模型推理)
- [模型支持列表](#模型支持列表)

______________________________________________________________________

[MMDetection3d](https://github.com/open-mmlab/mmdetection3d)，又称 `mmdet3d`， 是一个基于 PyTorch 的目标检测开源工具箱, 下一代面向3D检测的平台。它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

## 安装 mmdet3d

因为依赖的 codebase 不在 master 分支，所以要切到相应分支：

| codebase |  commit   |
| :------: | :-------: |
| mmdet3d  | v1.1.0rc1 |
|   mmcv   | v2.0.0rc1 |
|  mmdet   | v3.0.0rc1 |
|  mmseg   | v1.0.0rc0 |

先安装前置依赖 mmcv/mmdet/mmseg，再安装 mmdet3d

```bash
python3 -m pip install openmim --user
python3 -m mim install mmcv==2.0.0rc1 mmdet==3.0.0rc1 mmseg==1.0.0rc0 --user

git clone https://github.com/open-mmlab/mmdetection3d --branch v1.1.0rc1
cd mmdetection3d
python3 -m pip install .
cd -
```

成功后 `tools/check_env.py` 应能正常显示 mmdet3d 版本号。

```bash
python3 tools/check_env.py
..
11/11 13:56:19 - mmengine - INFO - **********Codebase information**********
11/11 13:56:19 - mmengine - INFO - mmdet:       3.0.0rc1
11/11 13:56:19 - mmengine - INFO - mmseg:       1.0.0rc0
..
11/11 13:56:19 - mmengine - INFO - mmdet3d:     1.1.0rc1
```

## 模型转换

使用 `tools/deploy.py` 把 mmdet3d 转到相应后端，以 centerpoint onnxruntime 为例：

```bash
export MODEL_CONFIG=/path/to/mmdetection3d/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py

export MODEL_PATH=https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210816_064624-0f3299c0.pth

export TEST_DATA=/path/to/mmdetection3d/tests/data/nuscenes/sweeps/LIDAR_TOP/n008-2018-09-18-12-07-26-0400__LIDAR_TOP__1537287083900561.pcd.bin

python3 tools/deploy.py configs/mmdet3d/voxel-detection/voxel-detection_onnxruntime_dynamic.py $MODEL_CONFIG $MODEL_PATH $TEST_DATA --work-dir centerpoint
```

`work-dir` 应生成对应 onnx

```bash
ls -lah centerpoint
..
-rw-rw-r--  1 rg rg  87M 11月  4 19:48 end2end.onnx
```

## 模型推理

目前 mmdet3d 的 voxelize 预处理和后处理未转成 onnx 操作；C++ SDK 也未实现 voxelize 计算。调用方需参照对应 [python 实现](../../../mmdeploy/codebase/mmdet3d/deploy/voxel_detection_model.py) 完成。

## 模型支持列表

|                                                                                  model                                                                                  | dataset  | onnxruntime | openvino | tensorrt\* |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------: | :---------: | :------: | :--------: |
| [centerpoint](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py) | nuScenes |     ✔️      |    ✔️    |     ✔️     |
|             [pointpillars](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py)              | nuScenes |     ✔️      |    ✔️    |     ✔️     |
|            [pointpillars](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py)            |  KITTI   |     ✔️      |    ✔️    |     ✔️     |

- 考虑到 ScatterND、动态 shape 等已知问题，请确保 trt >= 8.4
