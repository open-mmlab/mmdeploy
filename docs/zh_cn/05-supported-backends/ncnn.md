# ncnn 支持情况

目前对 ncnn 特性使用情况如下：

|      feature       | windows | linux | mac | android |
| :----------------: | :-----: | :---: | :-: | :-----: |
|   fp32 inference   |   ✔️    |  ✔️   | ✔️  |   ✔️    |
| int8 model convert |    -    |  ✔️   | ✔️  |    -    |
|    nchw layout     |   ✔️    |  ✔️   | ✔️  |   ✔️    |
|   Vulkan support   |    -    |  ✔️   | ✔️  |   ✔️    |

以下特性还不能由 mmdeploy 自动开启，需要手动修改 ncnn 编译参数、或在 SDK 中调整运行参数

- bf16 inference
- nc4hw4 layout
- profiling per layer
- 关闭 NCNN_STRING 以减小 so 体积
- 设置线程数和 CPU 亲和力
