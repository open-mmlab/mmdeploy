# MMDetection3d 模型部署

- [MMDetection3d 模型部署](#mmdetection3d-模型部署)
  - [安装 mmdet3d](#安装-mmdet3d)
  - [模型转换](#模型转换)
  - [模型推理](#模型推理)
  - [模型支持列表](#模型支持列表)

______________________________________________________________________

[MMDetection3d](https://github.com/open-mmlab/mmdetection3d)，又称 `mmdet3d`， 是一个基于 PyTorch 的目标检测开源工具箱, 下一代面向3D检测的平台。它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

## 安装 mmdet3d

我们可以通过 [mim](https://github.com/open-mmlab/mim) 来安装 mmdet3d.
更多安装方式可参考该[文档](https://mmdetection3d.readthedocs.io/en/latest/get_started.html#installation)

```bash
python3 -m pip install -U openmim
python3 -m mim install "mmdet3d>=1.1.0"
```

## 模型转换

使用 `tools/deploy.py` 把 mmdet3d 转到相应后端，以 centerpoint onnxruntime 为例：

```bash
# 切换到 mmdeploy 根目录
# 通过mim下载centerpoint模型
mim download mmdet3d --config centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d --dest .

export MODEL_CONFIG=centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py

export MODEL_PATH=centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth

export TEST_DATA=tests/data/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151612397179.pcd.bin

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

|                                                                                model                                                                                 |        task         | dataset  | onnxruntime | openvino | tensorrt\* |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------: | :------: | :---------: | :------: | :--------: |
| [centerpoint](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py) |   voxel detection   | nuScenes |     ✔️      |    ✔️    |     ✔️     |
|             [pointpillars](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py)              |   voxel detection   | nuScenes |     ✔️      |    ✔️    |     ✔️     |
|            [pointpillars](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py)            |   voxel detection   |  KITTI   |     ✔️      |    ✔️    |     ✔️     |
|                   [smoke](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/smoke/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py)                    | monocular detection |  KITTI   |     ✔️      |    x     |     ✔️     |

- 考虑到 ScatterND、动态 shape 等已知问题，请确保 trt >= 8.6
