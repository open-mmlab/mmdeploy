# 支持 SNPE

mmdeploy 集成 snpe 的方式简单且有效： Client/Server 模式。

这种模式

1. 能剥离`模型转换`和`推理`环境：

- 推理无关事项在算力更高的设备上完成；
- 推理计算，能拿到 gpu/npu 真实结果，而非 CPU 模拟器数值。

2. 能覆盖到成本敏感的设备。armv7/risc-v/mips 芯片满足产品需求，但往往对 Python 支持有限；

3. 能简化 mmdeploy 安装步骤。如果只想转 .dlc 模型测试精度，不需要接触 snpe tutorial。

## 一、部署推理服务

下载编译好的 snpe inference server 包， `adb push` 到手机，执行。

```bash
$ wget https://media.githubusercontent.com/media/tpoisonooo/mmdeploy-onnx2ncnn-testdata/main/snpe-inference-server-1.59.zip
$ unzip snpe-inference-server-1.59.zip
$ adb push snpe-inference-server-1.59  /data/local/tmp/
```

如果需要自己编译，可参照 [NDK 交叉编译 snpe inference sever](../06-appendix/cross-build-ndk-gRPC.md) 。

## 二、安装 mmdeploy

1. 环境要求

| 事项    | 版本               | 备注              |
| ------- | ------------------ | ----------------- |
| host OS | ubuntu18.04 x86_64 | snpe 工具指定版本 |
| Python  | 3.6.0              | snpe 工具指定版本 |

## 三、测试模型

## 四、编译 SDK
