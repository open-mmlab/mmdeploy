<div align="center">
  <img src="resources/mmdeploy-logo.png" width="600"/>
</div>

## Introduction

English | [简体中文](README_zh-CN.md)

MMDeploy is an open-source deep learning model deployment toolset. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

<div align="center">
  <img src="https://socialistmodernism.com/wp-content/uploads/2017/07/placeholder-image.png"/>
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

Please refer to [build.md](docs/en/build.md) for installation.

## Getting Started
Please see [getting_started.md](docs/en/get_started.md) for the basic usage of MMDeploy. We also provide other tutorials for:
- [how to convert model](docs/en/tutorials/how_to_convert_model.md)
- [how to write config](docs/en/tutorials/how_to_write_config.md)
- [how to support new models](docs/en/tutorials/how_to_support_new_models.md)
- [how to measure performance of models](docs/en/tutorials/how_to_measure_performance_of_models.md)
- [MMDeploy's SDK Model spec](docs/en/tutorials/sdk_model_spec.md)
- [how to integrate SDK to your application](docs/en/tutorials/sdk_integration.md)
- [how to develop postprocessing components in SDK](docs/en/tutorials/postprocess_component_development.md)


Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.


## Benchmark and model zoo

Results and supported model list are available in the [benchmark](docs/en/benchmark.md) and [model list](docs/en/tutorials/how_to_convert_model.md).

## Contributing

We appreciate all contributions to improve MMDeploy. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

We would like to thank OpenVINO team, for their remarkable efforts to export MMDetection models to OpenVINO and integrate OpenVINO into MMDeploy backends

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
