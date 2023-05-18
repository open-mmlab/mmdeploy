# 如何进行服务端部署

模型转换后，MMDeploy 提供基于 Triton Inference Server 的模型服务端部署。

## 支持的任务

目前支持以下任务：

- [image-classification](../../../demo/triton/image-classification/README.md)
- [instance-segmentation](../../../demo/triton/instance-segmentation)
- [keypoint-detection](../../../demo/triton/keypoint-detection)
- [object-detection](../../../demo/triton/object-detection)
- [oriented-object-detection](../../../demo/triton/oriented-object-detection)
- [semantic-segmentation](../../../demo/triton/semantic-segmentation)
- [text-detection](../../../demo/triton/text-detection)
- [text-recognition](../../../demo/triton/text-recognition)
- [text-ocr](../../../demo/triton/text-ocr)

## 如何部署 Triton 服务

为了使用 Triton Inference Server, 我们需要：

1. 编译 MMDeploy Triton Backend
2. 准备模型库(包括模型文件，以及配置文件)

### 编译 MMDeploy Triton Backend

a) 使用 Docker 镜像

为了方便使用，我们提供了 Docker 镜像，支持对通过 MMDeploy 转换的模型进行部署。镜像支持 Tensorrt 以及 ONNX Runtime 作为后端。若需要其他后端，可选择从源码进行编译。

b) 从源码编译

从源码编译 MMDeploy 的方式可参考[源码手动安装](../01-how-to-build/build_from_source.md)，要编译 MMDeploy Triton Backend，需要在编译命令中添加：`-DTRITON_MMDEPLOY_BACKEND=ON`。默认使用最新版本的 Triton Backend，若要使用旧版本的 Triton Backend，可在编译命令中添加`-DTRITON_TAG=r22.12`

### 准备模型库

Triton Inference Server 有一套自己的模型描述规则，通过 `tools/deploy.py ... --dump-info ` 转换的模型需要调整格式才能使 Triton 正确加载，我们为各任务准备了模版，可以运行 `demo/triton/to_triton_model.py` 转换脚本格式进行修改。完整的样例可参考各个 demo 的说明。
