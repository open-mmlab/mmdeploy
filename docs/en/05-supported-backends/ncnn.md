# Supported ncnn feature

The current use of the ncnn feature is as follows:

|      feature       | windows | linux | mac | android |
| :----------------: | :-----: | :---: | :-: | :-----: |
|   fp32 inference   |   ✔️    |  ✔️   | ✔️  |   ✔️    |
| int8 model convert |    -    |  ✔️   | ✔️  |    -    |
|    nchw layout     |   ✔️    |  ✔️   | ✔️  |   ✔️    |
|   Vulkan support   |    -    |  ✔️   | ✔️  |   ✔️    |

The following features cannot be automatically enabled by mmdeploy and you need to manually modify the ncnn build options or adjust the running parameters in the SDK

- bf16 inference
- nc4hw4 layout
- Profiling per layer
- Turn off NCNN_STRING to reduce .so file size
- Set thread number and CPU affinity
