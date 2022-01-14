<div align="center">
  <img src="resources/mmdeploy-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
</div>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdeploy.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdeploy/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdeploy/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdeploy/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdeploy)
[![license](https://img.shields.io/github/license/open-mmlab/mmdeploy.svg)](https://github.com/open-mmlab/mmdeploy/blob/master/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmdeploy.svg)](https://github.com/open-mmlab/mmdeploy/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdeploy.svg)](https://github.com/open-mmlab/mmdeploy/issues)

## 介绍

[English](README.md) | 简体中文

MMDeploy 是一个开源深度学习模型部署工具箱，它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

<div align="center">
  <img src="resources/introduction.png" width="800"/>
</div>

### 主要特性

- **全面支持 OpenMMLab 模型的部署**

  我们为 OpenMMLab 各算法库提供了统一的模型部署工具箱。已支持的算法库如下所示，未来将支持更多的算法库
  - [x] MMClassification
  - [x] MMDetection
  - [x] MMSegmentation
  - [x] MMEditing
  - [x] MMOCR

- **支持多种推理后端**

  模型可以导出为多种推理引擎文件，并在对应的后端上进行推理。 如下后端已经支持，后续将支持更多的后端。
  - [x] ONNX Runtime
  - [x] TensorRT
  - [x] PPLNN
  - [x] ncnn
  - [x] OpenVINO

- **高度可扩展的 SDK 开发框架 (C/C++)**

  SDK 中所有的组件均可扩展。比如用于图像处理的`Transform`，用于深度学习网络推理的`Net`，后处理中的`Module`等等。

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 安装

请参考[构建项目](https://mmdeploy.readthedocs.io/zh_CN/latest/build.html)进行安装。

## 快速入门

请参考[快速入门文档](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html)学习 MMDeploy 的基本用法。我们还提供了一些进阶教程，

- [如何进行模型转换](https://mmdeploy.readthedocs.io/zh_CN/latest/tutorials/how_to_convert_model.html)
- [如何编写配置文件](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_write_config.html)
- [如何支持新模型](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_support_new_models.html)
- [如何测试模型效果](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_measure_performance_of_models.html)

如果遇到问题，请参考 [常见问题解答](https://mmdeploy.readthedocs.io/zh_CN/latest/faq.html)。

## 基准与模型库

基准和支持的模型列表可以在[基准](https://mmdeploy.readthedocs.io/zh_CN/latest/benchmark.html)和[模型列表](https://mmdeploy.readthedocs.io/en/latest/supported_models.html)中获得。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMDeploy 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

由衷感谢以下团队为 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 做出的贡献：

- [OpenPPL](https://github.com/openppl-public)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)

## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 MMDeploy:

```BibTeX
@misc{=mmdeploy,
    title={OpenMMLab's Model Deployment Toolbox.},
    author={MMDeploy Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdeploy}},
    year={2021}
}
```

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
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/zhihu_qrcode.jpg" height="400" />
  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/qq_group_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
