# mmdeploy 各目录功能

本文主要介绍 mmdeploy 各目录功能，以及从模型到具体推理框架是怎么工作的。

## 一、大致看下目录结构

整个 mmdeploy 可以看成比较独立的两部分：模型转换 和 SDK。

我们介绍整个 repo 目录结构和功能，不必细究源码、有个印象即可。

外围目录功能：

```bash
$ cd /path/to/mmdeploy
$ tree -L 1
.
├── CMakeLists.txt    # 编译模型转换自定义算子和 SDK 的 cmake 配置
├── configs                   # 模型转换要用的算法库配置
├── csrc                          # SDK 和自定义算子
├── demo                      # 各语言的 ffi 接口应用实例，如 csharp、java、python 等
├── docker                   #  docker build
├── mmdeploy           # 用于模型转换的 python 包
├── requirements      # python 包安装依赖
├── service                    # 有些小板子不能跑 python，模型转换用的 C/S 模式。这个目录放 Server
├── tests                         # 单元测试
├── third_party           # SDK 和 ffi 要的第三方依赖
└── tools                        # 工具，也是一切功能的入口。除了 deploy.py 还有 onnx2xx.py、profiler.py 和 test.py
```

这样大致应该清楚了

- 模型转换主要看 tools + mmdeploy + 小部分 csrc 目录；
- 而 SDK 的本体在 csrc + third_party + demo 三个目录。

## 二、模型转换

模型以 mmpretrain 的 ViT 为例，推理框架就用 ncnn 举例。其他模型、推理都是类似的。

我们看下 mmdeploy/mmdeploy 目录结构，有个印象即可：

```bash
.
├── apis                             # tools 工具用的 api，都是这里实现的，如 onnx2ncnn.py
│   ├── calibration.py          # trt 专用收集量化数据
│   ├── core                              # 软件脚手架
│   ├── extract_model.py  # onnx 模型只想导出一部分，切 onnx 用的
│   ├── inference.py             # 抽象函数，实际会调 torch/ncnn 具体的 inference
│   ├── ncnn                            # 引用 backend/ncnn 的函数，只是包了一下
│   └── visualize.py              # 还是抽象函数，实际会调用 torch/ncnn 具体的 inference 和 visualize
..
├── backend                  # 具体的 backend 包装
│   ├── base                            # 因为有多个 backend，所以得有个 base 类的 OO 设计
│   ├── ncnn                           # 这里为模型转换调用 ncnn python 接口
│   │   ├── init_plugins.py           # 找 ncnn 自定义算子和 ncnn 工具的路径
│   │   ├── onnx2ncnn.py            # 把 `mmdeploy_onnx2ncnn` 封装成 python 接口
│   │   ├── quant.py                       # 封装 `ncnn2int8` 工具为 python 接口
│   │   └── wrapper.py                  # 封装 pyncnn forward 接口
..
├── codebase                #  mm 系列算法 forward 重写
│   ├── base                          # 有多个算法，需要点 OO 设计
│   ├── mmpretrain                      #  mmpretrain 相关模型重写
│   │   ├── deploy                       # mmpretrain 对 base 抽象 task/model/codebase 的实现
│   │   └── models                      # 开始真正的模型重写
│   │       ├── backbones                 # 骨干网络部分的重写，例如  multiheadattention
│   │       ├── heads                           # 例如  MultiLabelClsHead
│   │       ├── necks                            # 例如 GlobalAveragePooling
│..
├── core                         # 软件脚手架，重写机制怎么实现的
├── mmcv                     # mmcv 有的 opr 也需要重写
├── pytorch                 #  针对 ncnn 重写 torch 的 opr，例如 Gemm
..
```

上面的每一行是需要读的，请勿跳过。

当敲下`tools/deploy.py` 转换 ViT，核心是这 3 件事：

1. mmpretrain ViT forward 过程的重写
2. ncnn 不支持 gather opr，自定义一下、和 libncnn.so 一起加载
3. 真实跑一遍，渲染结果，确保正确

### 1. forward 重写

因为 onnx 会生成稀碎的算子、ncnn 也不是完美支持 onnx，所以 mmdeploy 的方案是劫持有问题的 forward 代码、改成适合 ncnn 的 onnx 结果。

例如把 `conv -> shape -> concat_const -> reshape` 过程改成 `conv -> reshape`，削掉多余的 `shape` 和 `concat` 算子。

所有的 mmpretrain 算法重写都在 `mmdeploy/codebase/mmpretrain/models`目录。

### 2. 自定义算子

针对 ncnn 自定义的算子都在 `csrc/mmdeploy/backend_ops/ncnn/`目录，编译后和 libncnn.so 一起加载。本质是在 hotfix ncnn，目前实现了

- topk
- tensorslice
- shape
- gather
- expand
- constantofshape

### 3. 转换和测试

ncnn 的兼容性较好，转换用的是修改后的 `mmdeploy_onnx2ncnn`，推理封装了 `pyncnn`+ 自定义 ops。

遇到 snpe 这种不支持 python 的框架，则使用 C/S 模式：用 gRPC 等协议封装一个 server，转发真实的推理结果。

渲染使用上游算法框架的渲染 API，mmdeploy 自身不做绘制。

## 三、SDK

模型转换完成后，可用 C++ 编译的 SDK 执行在不同平台上。

我们看下 csrc/mmdeploy 目录结构：

```bash
.
├── apis           # Csharp、java、go、Rust 等 ffi 接口
├── backend_ops    # 各推理框架的自定义算子
├── CMakeLists.txt
├── codebase       # 各 mm 算法框架偏好的结果类型，例如检测任务多用 bbox
├── core           # 脚手架，对图、算子、设备的抽象
├── device         # CPU/GPU device 抽象的实现
├── execution      # 对 exec 抽象的实现
├── graph          # 对图抽象的实现
├── model          # 实现 zip 压缩和非压缩两种工作目录
├── net            # net 的具体实现，例如封装了 ncnn forward C 接口
├── preprocess     # 预处理的实现
└── utils          # OCV 工具类
```

SDK 本质是设计了一套计算图的抽象，把**多个模型**的

- 预处理
- 推理
- 后处理

调度起来，同时提供多种语言的 ffi。
