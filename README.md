<div align="center">
  <img src="resources/mmdeploy-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
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

## Introduction

English | [简体中文](README_zh-CN.md)

MMDeploy is an open-source deep learning model deployment toolset. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

<div align="center">
  <img src="resources/introduction.png" width="800"/>
</div>


### Major features

- **Fully support OpenMMLab models**

  We provide a unified model deployment toolbox for the codebases in OpenMMLab. The supported codebases are listed as below, and more will be added in the future
  - [x] MMClassification
  - [x] MMDetection
  - [x] MMSegmentation
  - [x] MMEditing
  - [x] MMOCR

- **Multiple inference backends are available**

  Models can be exported and run in different backends. The following ones are supported, and more will be taken into consideration
  - [x] ONNX Runtime
  - [x] TensorRT
  - [x] PPLNN
  - [x] ncnn
  - [x] OpenVINO

- **Efficient and highly scalable SDK Framework by C/C++**

    All kinds of modules in SDK can be extensible, such as `Transform` for image processing, `Net` for Neural Network inference, `Module` for postprocessing and so on

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation

Please refer to [build.md](https://mmdeploy.readthedocs.io/en/latest/build.html) for installation.

## Getting Started

Please see [getting_started.md](https://mmdeploy.readthedocs.io/en/latest/get_started.html) for the basic usage of MMDeploy. We also provide other tutorials for:

- [how to convert model](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_convert_model.html)
- [how to write config](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_write_config.html)
- [how to support new models](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_support_new_models.html)
- [how to measure performance of models](https://mmdeploy.readthedocs.io/en/latest/tutorials/how_to_measure_performance_of_models.html)

Please refer to [FAQ](https://mmdeploy.readthedocs.io/en/latest/faq.html) for frequently asked questions.

## Benchmark and model zoo

Results and supported model list are available in the [benchmark](https://mmdeploy.readthedocs.io/en/latest/benchmark.html) and [model list](https://mmdeploy.readthedocs.io/en/latest/supported_models.html).

## Contributing

We appreciate all contributions to improve MMDeploy. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

We would like to sincerely thank the following teams for their contributions to [MMDeploy](https://github.com/open-mmlab/mmdeploy):
- [OpenPPL](https://github.com/openppl-public)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{=mmdeploy,
    title={OpenMMLab's Model Deployment Toolbox.},
    author={MMDeploy Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdeploy}},
    year={2021}
}
```

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
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab Human Pose and Shape Estimation Toolbox and Benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning Toolbox and Benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab Model Compression Toolbox and Benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
