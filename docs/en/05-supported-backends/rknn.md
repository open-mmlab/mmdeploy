# Supported RKNN feature

Currently, MMDeploy only tests rk3588 with linux platform.

The following features cannot be automatically enabled by mmdeploy and you need to manually modify the configuration in MMDeploy like [here](https://github.com/open-mmlab/mmdeploy/blob/master/configs/_base_/backends/rknn.py).

- target_platform other than `3588`
- quantization settings
- optimization level other than 3
