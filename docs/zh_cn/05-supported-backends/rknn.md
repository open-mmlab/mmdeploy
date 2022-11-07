# 支持的 RKNN 特征

目前, MMDeploy 只在 rk3588 和 rv1126 的 linux 平台上测试过.

以下特性需要手动在 MMDeploy 自行配置，如[这里](https://github.com/open-mmlab/mmdeploy/blob/master/configs/_base_/backends/rknn.py).

- target_platform ！= default
- quantization settings
- optimization level ！= 1
