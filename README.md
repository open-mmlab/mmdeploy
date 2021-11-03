## Introduction

English | [简体中文](README_zh-CN.md)

MMDeploy is an open-source deep learning model deployment toolset. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

### Major features

- **OpenMMLab model support**

  Models in OpenMMLab can be deployed with this project. Such as MMClassification, MMDetection, etc.

- **Multiple inference engine support**

  Models can be exported and run in different backends. Such as ONNX Runtime, TensorRT, etc.

- **Model rewrite**

  Modules and functions used in models can be rewritten to meet the demand of different backends. It is easy to add new model support.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Codebase and Backend support

Supported codebase:

- [x] MMClassification
- [x] MMDetection
- [x] MMSegmentation
- [x] MMEditing
- [x] MMOCR

Supported backend:

- [x] ONNX Runtime
- [x] TensorRT
- [x] PPL
- [x] ncnn
- [x] OpenVINO

## Installation

Please refer to [build.md](docs/build.md) for installation.

## Getting Started

Please read [how_to_convert_model.md](docs/tutorials/how_to_convert_model.md) for the basic usage of MMDeploy. There are also tutorials for [how to write config](docs/tutorials/how_to_write_config.md), [how to support new models](docs/tutorials/how_to_support_new_models.md) and [how to test model](docs/tutorials/how_to_test_model.md).

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{=mmdeploy,
    title={OpenMMLab's Model deployment toolbox.},
    author={MMDeploy Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdeploy}},
    year={2021}
}
```

## Contributing

We appreciate all contributions to improve MMDeploy. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
