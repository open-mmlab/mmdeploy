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

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdeploy.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdeploy/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdeploy/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdeploy/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdeploy)
[![license](https://img.shields.io/github/license/open-mmlab/mmdeploy.svg)](https://github.com/open-mmlab/mmdeploy/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/open-mmlab/mmdeploy)](https://github.com/open-mmlab/mmdeploy/issues)
[![open issues](https://img.shields.io/github/issues-raw/open-mmlab/mmdeploy)](https://github.com/open-mmlab/mmdeploy/issues)

English | [简体中文](README_zh-CN.md)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218352562-cdded397-b0f3-4ca1-b8dd-a60df8dca75b.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
</div>

## Highlights

The MMDeploy 1.x has been released, which is adapted to upstream codebases from OpenMMLab 2.0. Please **align the version** when using it.
The default branch has been switched to `main` from `master`. MMDeploy 0.x (`master`) will be deprecated and new features will only be added to MMDeploy 1.x (`main`) in future.

| mmdeploy | mmengine |   mmcv   |  mmdet   | others |
| :------: | :------: | :------: | :------: | :----: |
|  0.x.y   |    -     | \<=1.x.y | \<=2.x.y | 0.x.y  |
|  1.x.y   |  0.x.y   |  2.x.y   |  3.x.y   | 1.x.y  |

[deploee](https://platform.openmmlab.com/deploee/) offers over 2,300 AI models in ONNX, NCNN, TRT and OpenVINO formats. Featuring a built-in list of real hardware devices, deploee enables users to convert Torch models into any target inference format for profiling purposes.

## Introduction

MMDeploy is an open-source deep learning model deployment toolset. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

<div align="center">
  <img src="resources/introduction.png">
</div>

## Main features

### Fully support OpenMMLab models

The currently supported codebases and models are as follows, and more will be included in the future

- [mmpretrain](docs/en/04-supported-codebases/mmpretrain.md)
- [mmdet](docs/en/04-supported-codebases/mmdet.md)
- [mmseg](docs/en/04-supported-codebases/mmseg.md)
- [mmagic](docs/en/04-supported-codebases/mmagic.md)
- [mmocr](docs/en/04-supported-codebases/mmocr.md)
- [mmpose](docs/en/04-supported-codebases/mmpose.md)
- [mmdet3d](docs/en/04-supported-codebases/mmdet3d.md)
- [mmrotate](docs/en/04-supported-codebases/mmrotate.md)
- [mmaction2](docs/en/04-supported-codebases/mmaction2.md)

### Multiple inference backends are available

The supported Device-Platform-InferenceBackend matrix is presented as following, and more will be compatible.

The benchmark can be found from [here](docs/en/03-benchmark/benchmark.md)

<div style="width: fit-content; margin: auto;">
<table>
  <tr>
    <th>Device / <br> Platform</th>
    <th>Linux</th>
    <th>Windows</th>
    <th>macOS</th>
    <th>Android</th>
  </tr>
  <tr>
    <th>x86_64 <br> CPU</th>
    <td>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-ort.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-ort.yml"></a></sub> <sub>onnxruntime</sub> <br>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-pplnn.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-pplnn.yml"></a></sub> <sub>pplnn</sub> <br>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-ncnn.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-ncnn.yml"></a></sub> <sub>ncnn</sub> <br>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-torchscript.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-torchscript.yml"></a></sub> <sub>LibTorch</sub> <br>
        <sub><img src="https://img.shields.io/badge/build-no%20status-lightgrey"></sub> <sub>OpenVINO</sub> <br>
        <sub><img src="https://img.shields.io/badge/build-no%20status-lightgrey"></sub> <sub>TVM</sub> <br>
    </td>
    <td>
        <sub><img src="https://img.shields.io/badge/build-no%20status-lightgrey"></sub> <sub>onnxruntime</sub> <br>
        <sub><img src="https://img.shields.io/badge/build-no%20status-lightgrey"></sub> <sub>OpenVINO</sub> <br>
        <sub><img src="https://img.shields.io/badge/build-no%20status-lightgrey"></sub> <sub>ncnn</sub> <br>
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
  </tr>

<tr>
    <th>ARM <br> CPU</th>
    <td>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/build.yml"><img src="https://byob.yarr.is/open-mmlab/mmdeploy/cross_build_aarch64"></a></sub> <sub>ncnn</sub> <br>
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-ncnn.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-ncnn.yml"></a></sub> <sub>ncnn</sub> <br>
    </td>
  </tr>

<tr>
    <th>RISC-V</th>
    <td>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/linux-riscv64-gcc.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/linux-riscv64-gcc.yml"></a></sub> <sub>ncnn</sub> <br>
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
  </tr>

<tr>
    <th>NVIDIA <br> GPU</th>
    <td>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/build.yml"><img src="https://byob.yarr.is/open-mmlab/mmdeploy/build_cuda113_linux"></a></sub> <sub>onnxruntime</sub> <br>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/build.yml"><img src="https://byob.yarr.is/open-mmlab/mmdeploy/build_cuda113_linux"></a></sub> <sub>TensorRT</sub> <br>
        <sub><img src="https://img.shields.io/badge/build-no%20status-lightgrey"></sub> <sub>LibTorch</sub> <br>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-pplnn.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-pplnn.yml"></a></sub> <sub>pplnn</sub> <br>
    </td>
    <td>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/build.yml"><img src="https://byob.yarr.is/open-mmlab/mmdeploy/build_cuda113_windows"></a></sub> <sub>onnxruntime</sub> <br>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/build.yml"><img src="https://byob.yarr.is/open-mmlab/mmdeploy/build_cuda113_windows"></a></sub> <sub>TensorRT</sub> <br>
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
  </tr>

<tr>
    <th>NVIDIA <br> Jetson</th>
    <td>
        <sub><img src="https://img.shields.io/badge/build-no%20status-lightgrey"></sub> <sub>TensorRT</sub> <br>
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
  </tr>

<tr>
    <th>Huawei <br> ascend310</th>
    <td>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-ascend.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-ascend.yml"></a></sub> <sub>CANN</sub> <br>
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
  </tr>

<tr>
    <th>Rockchip</th>
    <td>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-rknn.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-rknn.yml"></a></sub> <sub>RKNN</sub> <br>
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
  </tr>

<tr>
    <th>Apple M1</th>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
    <td>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-coreml.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-coreml.yml"></a></sub> <sub>CoreML</sub> <br>
    </td>
    <td align="center">
        -
    </td>
  </tr>

<tr>
    <th>Adreno <br> GPU</th>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
    <td>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-snpe.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-snpe.yml"></a></sub> <sub>SNPE</sub> <br>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-ncnn.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-ncnn.yml"></a></sub> <sub>ncnn</sub> <br>
    </td>
  </tr>

<tr>
    <th>Hexagon <br> DSP</th>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
    <td align="center">
        -
    </td>
    <td>
        <sub><a href="https://github.com/open-mmlab/mmdeploy/actions/workflows/backend-snpe.yml"><img src="https://img.shields.io/github/actions/workflow/status/open-mmlab/mmdeploy/backend-snpe.yml"></a></sub> <sub>SNPE</sub> <br>
    </td>
  </tr>
</table>
</div>

### Efficient and scalable C/C++ SDK Framework

All kinds of modules in the SDK can be extended, such as `Transform` for image processing, `Net` for Neural Network inference, `Module` for postprocessing and so on

## [Documentation](https://mmdeploy.readthedocs.io/en/latest/)

Please read [getting_started](docs/en/get_started.md) for the basic usage of MMDeploy. We also provide tutoials about:

- [Build](docs/en/01-how-to-build/build_from_source.md)
  - [Build from Docker](docs/en/01-how-to-build/build_from_docker.md)
  - [Build from Script](docs/en/01-how-to-build/build_from_script.md)
  - [Build for Linux](docs/en/01-how-to-build/linux-x86_64.md)
  - [Build for macOS](docs/en/01-how-to-build/macos-arm64.md)
  - [Build for Win10](docs/en/01-how-to-build/windows.md)
  - [Build for Android](docs/en/01-how-to-build/android.md)
  - [Build for Jetson](docs/en/01-how-to-build/jetsons.md)
  - [Build for SNPE](docs/en/01-how-to-build/snpe.md)
  - [Cross Build for aarch64](docs/en/01-how-to-build/cross_build_ncnn_aarch64.md)
- User Guide
  - [How to convert model](docs/en/02-how-to-run/convert_model.md)
  - [How to write config](docs/en/02-how-to-run/write_config.md)
  - [How to profile model](docs/en/02-how-to-run/profile_model.md)
  - [How to quantize model](docs/en/02-how-to-run/quantize_model.md)
  - [Useful tools](docs/en/02-how-to-run/useful_tools.md)
- Developer Guide
  - [Architecture](docs/en/07-developer-guide/architecture.md)
  - [How to support new models](docs/en/07-developer-guide/support_new_model.md)
  - [How to support new backends](docs/en/07-developer-guide/support_new_backend.md)
  - [How to partition model](docs/en/07-developer-guide/partition_model.md)
  - [How to test rewritten model](docs/en/07-developer-guide/test_rewritten_models.md)
  - [How to test backend ops](docs/en/07-developer-guide/add_backend_ops_unittest.md)
  - [How to do regression test](docs/en/07-developer-guide/regression_test.md)
- Custom Backend Ops
  - [ncnn](docs/en/06-custom-ops/ncnn.md)
  - [ONNXRuntime](docs/en/06-custom-ops/onnxruntime.md)
  - [tensorrt](docs/en/06-custom-ops/tensorrt.md)
- [FAQ](docs/en/faq.md)
- [Contributing](.github/CONTRIBUTING.md)

## Benchmark and Model zoo

You can find the supported models from [here](docs/en/03-benchmark/supported_models.md) and their performance in the [benchmark](docs/en/03-benchmark/benchmark.md).

## Contributing

We appreciate all contributions to MMDeploy. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

We would like to sincerely thank the following teams for their contributions to [MMDeploy](https://github.com/open-mmlab/mmdeploy):

- [OpenPPL](https://github.com/openppl-public)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)
- [ncnn](https://github.com/Tencent/ncnn)

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@misc{=mmdeploy,
    title={OpenMMLab's Model Deployment Toolbox.},
    author={MMDeploy Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdeploy}},
    year={2021}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMPretrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab pre-training toolbox and benchmark.
- [MMagic](https://github.com/open-mmlab/mmagic): Open**MM**Lab **A**dvanced, **G**enerative and **I**ntelligent **C**reation toolbox.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [Playground](https://github.com/open-mmlab/playground): A central hub for gathering and showcasing amazing projects built upon OpenMMLab.
