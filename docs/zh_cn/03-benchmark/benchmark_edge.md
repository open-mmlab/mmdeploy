# 边、端设备测试结果

这里给出我们边、端设备的测试结论，用户可以直接通过 [model profiling](../02-how-to-run/profile_model.md) 获得自己环境的结果。

## 软硬件环境

- host OS ubuntu 18.04
- backend SNPE-1.59
- device Mi11 (qcom 888)

## mmpretrain 模型

|                                                          model                                                           |   dataset   | spatial | fp32 top-1 (%) | snpe gpu hybrid fp32 top-1 (%) | latency (ms) |
| :----------------------------------------------------------------------------------------------------------------------: | :---------: | :-----: | :------------: | :----------------------------: | :----------: |
| [ShuffleNetV2](https://github.com/open-mmlab/mmpretrain/blob/main/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py) | ImageNet-1k | 224x224 |     69.55      |            69.83\*             |     20±7     |
|    [MobilenetV2](https://github.com/open-mmlab/mmpretrain/blob/main/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py)     | ImageNet-1k | 224x224 |     71.86      |            72.14\*             |     15±6     |

tips:

1. ImageNet-1k 数据集较大，仅使用一部分测试（8000/50000）
2. 边、端设备发热会降频，因此耗时实际上会波动。这里给出运行一段时间后、稳定的数值。这个结果更贴近实际需求

## mmocr 检测

|                                                         model                                                          |  dataset  | spatial  | fp32 hmean | snpe gpu hybrid hmean | latency(ms) |
| :--------------------------------------------------------------------------------------------------------------------: | :-------: | :------: | :--------: | :-------------------: | :---------: |
| [PANet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_resnet18_fpem-ffm_600e_icdar2015.py) | ICDAR2015 | 1312x736 |   0.795    |    0.785 @thr=0.9     |  3100±100   |

## mmpose 模型

|                                                                                  model                                                                                  |  dataset   | spatial | snpe hybrid AR@IoU=0.50 | snpe hybrid AP@IoU=0.50 | latency(ms) |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :-----: | :---------------------: | :---------------------: | :---------: |
| [pose_hrnet_w32](https://github.com/open-mmlab/mmpose/blob/main/configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py) | Animalpose | 256x256 |          0.997          |          0.989          |   630±50    |

tips:

- 测试 pose_hrnet 用的是 AnimalPose 的 test dataset，而非 val dataset

## mmseg

|                                                        model                                                         |  dataset   | spatial  | mIoU  | latency(ms) |
| :------------------------------------------------------------------------------------------------------------------: | :--------: | :------: | :---: | :---------: |
| [fcn](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/fcn/fcn_r18-d8_4xb2-80k_cityscapes-512x1024.py) | Cityscapes | 512x1024 | 71.11 |  4915±500   |

tips:

- fcn 用 512x1024 尺寸运行正常。Cityscapes 数据集 1024x2048 分辨率会导致设备重启

## 其他模型

- mmdet 需要手动把模型拆成两部分。因为
  - snpe 源码中 `onnx_to_ir.py` 仅能解析输入，`ir_to_dlc.py` 还不支持 topk
  - UDO （用户自定义算子）无法和 `snpe-onnx-to-dlc` 配合使用
- mmagic 模型
  - srcnn 需要 cubic resize，snpe 不支持
  - esrgan 可正常转换，但加载模型会导致设备重启
- mmrotate 依赖 [e2cnn](https://pypi.org/project/e2cnn/) ，需要手动安装 [其 Python3.6
  兼容分支](https://github.com/QUVA-Lab/e2cnn)
