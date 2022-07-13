# 边、端设备测试结果

这里给出我们边、端设备的测试结论，用户可以直接通过 [model profiling](../02-how-to-run/profile_model.md) 获得自己环境的结果。

## 软硬件环境

- host OS ubuntu 18.04
- backend SNPE-1.59
- device Mi11 (qcom 888)

## 分类任务

|                                                              model                                                               |   dataset   | spatial | fp32 top-1 (%) | snpe gpu hybrid fp32 top-1 (%) | latency (ms) |
| :------------------------------------------------------------------------------------------------------------------------------: | :---------: | :-----: | :------------: | :-----------------------: | :----------: |
| [ShuffleNetV2](https://github.com/open-mmlab/mmclassification/blob/master/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py) | ImageNet-1k | 224x224 |     69.55      |          69.83\*          |     20±7     |
|    [MobilenetV2](https://github.com/open-mmlab/mmclassification/blob/master/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py)     | ImageNet-1k | 224x224 |     71.86      |          72.14\*          |     15±6     |

## ocr 检测任务

|                                                       model                                                       |  dataset  | spatial | fp32 hmean |   snpe gpu hybrid hmean   | latency(ms)
| :---------------------------------------------------------------------------------------------------------------: | :-------: |  :-------: | :--------: | :------------: | :------------: |
| [PANet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py) |  ICDAR2015 | 1312x736 |  0.795    | 0.785 @thr=0.9 | 3100±100

## 说明

1. ImageNet-1k 数据集较大，仅使用一部分测试（8000/50000）

2. 边、端设备发热会降频，因此耗时实际上会波动。这里给出运行一段时间后、稳定的数值。这个结果更贴近实际需求。
