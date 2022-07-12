# 边、端设备测试结果

这里给出我们边、端设备的测试结论，用户可以直接通过 [model profiling](../02-how-to-run/profile_model.md) 获得自己环境的结果。

## 软硬件环境

- host OS ubuntu 18.04
- backend SNPE-1.59
- device Mi11 (qcom 888)

## 测试结果

| model     |   dataset | spatial  | fp32 top-1 (%) | snpe gpu fp16-fp32 hybrid | latency (ms) |
| :--------------------------------------------------------------------------------------------------------------------------: | :---------: |  :---------: | :------------: | :------------: | :------------: |
| [ShuffleNetV2](https://github.com/open-mmlab/mmclassification/blob/master/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py) | ImageNet-1k | 224x224 | 69.55 | 69.83\* | 20±7 |
| [Resnet-18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb16_cifar10.py) | Cifar10 | 600x400 | 94.82 | \* | 1020±10 |


## 说明

1. 数据集较大，仅使用 ImageNet-1k 的一部分测试（8000/50000）

2. 边、端设备发热会降频，因此耗时实际上会波动。这里给出运行一段时间后、稳定的数值。这个结果更贴近实际需求。
