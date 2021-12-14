## 介绍

[English](README.md) | 简体中文

MMDeploy 是一个开源深度学习模型部署工具箱，它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

### 主要特性

- **支持OpenMMLab模型的部署**

  可以使用本项目进行OpenMMLab的模型部署，比如 MMClassification，MMDetection 等等。

- **支持各类推理引擎**

  模型可以被导出并在各种推理引擎上进行推理，比如 ONNX Runtime， TensorRT 等等。

- **模型改写**

  模型中的模块与函数可以被改写以满足各种推理引擎的需求，便于添加新的模型部署需求。

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 已支持的算法库与推理引擎

支持的算法库：

- [x] MMClassification
- [x] MMDetection
- [x] MMSegmentation
- [x] MMEditing
- [x] MMOCR

支持的推理引擎:

- [x] ONNX Runtime
- [x] TensorRT
- [x] PPLNN
- [x] ncnn

## 安装

请参考[构建项目](docs/build.md)进行安装。

## 快速入门

请阅读 [如何进行模型转换](docs/tutorials/how_to_convert_model.md) 来了解基本的 MMDeploy 使用。

我们还提供了诸如 [如何编写配置文件](docs/tutorials/how_to_write_config.md)， [如何添加新模型支持](docs/tutorials/how_to_support_new_models.md) 和 [如何测试模型效果](docs/tutorials/how_to_measure_performance_of_models.md) 等教程。

如果遇到问题，请参考 [常见问题解答](docs/faq.md)。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
