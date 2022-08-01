# 如何进行回归测试

<!-- -->

这篇教程介绍了如何进行回归测试。部署配置文件由`每个codebase的回归配置文件`，`推理框架配置信息`组成。

<!-- TOC -->

- [如何进行回归测试](#如何进行回归测试)
  - [1. 环境搭建](#1-环境搭建)
    - [MMDeploy的安装及配置](#mmdeploy的安装及配置)
    - [Python环境依赖](#python环境依赖)
  - [2. 用法](#2-用法)
    - [参数解析](#参数解析)
    - [注意事项](#注意事项)
  - [例子](#例子)
  - [3. 回归测试配置文件](#3-回归测试配置文件)
    - [示例及参数解析](#示例及参数解析)
  - [4. 生成的报告](#4-生成的报告)
    - [模板](#模板)
    - [示例](#示例)
  - [5. 支持的后端](#5-支持的后端)
  - [6. 支持的Codebase及其Metric](#6-支持的codebase及其metric)
  - [7. 注意事项](#7-注意事项)
  - [8. 常见问题](#8-常见问题)

<!-- TOC -->

## 1. 环境搭建

### MMDeploy的安装及配置

本章节的内容，需要提前根据[build 文档](../01-how-to-build/build_from_source.md)将 MMDeploy 安装配置好之后，才能进行。

### Python环境依赖

需要安装 test 的环境

```shell
pip install -r requirements/tests.txt
```

如果在使用过程是 numpy 报错，则更新一下 numpy

```shell
pip install -U numpy
```

## 2. 用法

```shell
python ./tools/regression_test.py \
    --codebase "${CODEBASE_NAME}" \
    --backends "${BACKEND}" \
    [--models "${MODELS}"] \
    --work-dir "${WORK_DIR}" \
    --device "${DEVICE}" \
    --log-level INFO \
    [--performance 或 -p] \
    [--checkpoint-dir "$CHECKPOINT_DIR"]
```

### 参数解析

- `--codebase` : 需要测试的 codebase，eg.`mmdet`, 测试多个 `mmcls mmdet ...`
- `--backends` : 筛选测试的后端, 默认测全部`backend`, 也可传入若干个后端，例如 `onnxruntime tesnsorrt`。如果需要一同进行 SDK 的测试，需要在 `tests/regression/${codebase}.yml` 里面的 `sdk_config` 进行配置。
- `--models` : 指定测试的模型, 默认测试 `yml` 中所有模型, 也可传入若干个模型名称，模型名称可参考相关yml配置文件。例如 `ResNet SE-ResNet "Mask R-CNN"`。注意的是，可传入只有字母和数字组成模型名称，例如 `resnet seresnet maskrcnn`。
- `--work-dir` : 模型转换、报告生成的路径，默认是`../mmdeploy_regression_working_dir`，注意路径中不要不含空格等特殊字符。
- `--checkpoint-dir`: PyTorch 模型文件下载保存路径，默认是`../mmdeploy_checkpoints`，注意路径中不要不含空格等特殊字符。
- `--device` : 使用的设备，默认 `cuda`。
- `--log-level` : 设置日记的等级，选项包括`'CRITICAL'， 'FATAL'， 'ERROR'， 'WARN'， 'WARNING'， 'INFO'， 'DEBUG'， 'NOTSET'`。默认是`INFO`。
- `-p` 或 `--performance` : 是否测试精度，加上则测试转换+精度，不加上则只测试转换

### 注意事项

对于 Windows 用户：

1. 要在 shell 命令中使用 `&&` 连接符，需要下载并使用 `PowerShell 7 Preview 5+`。
2. 如果您使用 conda env，可能需要在 regression_test.py 中将 `python3` 更改为 `python`，因为 `%USERPROFILE%\AppData\Local\Microsoft\WindowsApps` 目录中有 `python3.exe`。

## 例子

1. 测试 mmdet 和 mmpose 的所有 backend 的 **转换+精度**

```shell
python ./tools/regression_test.py \
    --codebase mmdet mmpose \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO \
    --performance
```

2. 测试 mmdet 和 mmpose 的某几个 backend 的 **转换+精度**

```shell
python ./tools/regression_test.py \
    --codebase mmdet mmpose \
    --backends onnxruntime tensorrt \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO \
    -p
```

3. 测试 mmdet 和 mmpose 的某几个 backend，**只测试转换**

```shell
python ./tools/regression_test.py \
    --codebase mmdet mmpose \
    --backends onnxruntime tensorrt \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO
```

4. 测试 mmdet 和 mmcls 的某几个 models，**只测试转换**

```shell
python ./tools/regression_test.py \
    --codebase mmdet mmpose \
    --models ResNet SE-ResNet "Mask R-CNN" \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO
```

## 3. 回归测试配置文件

### 示例及参数解析

```yaml
globals:
  codebase_dir: ../mmocr # 回归测试的 codebase 路径
  checkpoint_force_download: False # 回归测试是否重新下载模型即使其已经存在
  images: # 测试使用图片
    img_densetext_det: &img_densetext_det ../mmocr/demo/demo_densetext_det.jpg
    img_demo_text_det: &img_demo_text_det ../mmocr/demo/demo_text_det.jpg
    img_demo_text_ocr: &img_demo_text_ocr ../mmocr/demo/demo_text_ocr.jpg
    img_demo_text_recog: &img_demo_text_recog ../mmocr/demo/demo_text_recog.jpg
  metric_info: &metric_info # 指标参数
    hmean-iou: # 命名根据 metafile.Results.Metrics
      eval_name: hmean-iou # 命名根据 test.py --metrics args 入参名称
      metric_key: 0_hmean-iou:hmean # 命名根据 eval 写入 log 的 key name
      tolerance: 0.1 # 容忍的阈值区间
      task_name: Text Detection # 命名根据模型 metafile.Results.Task
      dataset: ICDAR2015 #命名根据模型 metafile.Results.Dataset
    word_acc: # 同上
      eval_name: acc
      metric_key: 0_word_acc_ignore_case
      tolerance: 0.2
      task_name: Text Recognition
      dataset: IIIT5K
  convert_image_det: &convert_image_det # det转换会使用到的图片
    input_img: *img_densetext_det
    test_img: *img_demo_text_det
  convert_image_rec: &convert_image_rec
    input_img: *img_demo_text_recog
    test_img: *img_demo_text_recog
  backend_test: &default_backend_test True # 是否对 backend 进行精度测试
  sdk: # SDK 配置文件
    sdk_detection_dynamic: &sdk_detection_dynamic configs/mmocr/text-detection/text-detection_sdk_dynamic.py
    sdk_recognition_dynamic: &sdk_recognition_dynamic configs/mmocr/text-recognition/text-recognition_sdk_dynamic.py

onnxruntime:
  pipeline_ort_recognition_static_fp32: &pipeline_ort_recognition_static_fp32
    convert_image: *convert_image_rec # 转换过程中使用的图片
    backend_test: *default_backend_test # 是否进行后端测试，存在则判断，不存在则视为 False
    sdk_config: *sdk_recognition_dynamic # 是否进行SDK测试，存在则使用特定的 SDK config 进行测试，不存在则视为不进行 SDK 测试
    deploy_config: configs/mmocr/text-recognition/text-recognition_onnxruntime_static.py # 使用的 deploy cfg 路径，基于 mmdeploy 的路径

  pipeline_ort_recognition_dynamic_fp32: &pipeline_ort_recognition_dynamic_fp32
    convert_image: *convert_image_rec
    backend_test: *default_backend_test
    sdk_config: *sdk_recognition_dynamic
    deploy_config: configs/mmocr/text-recognition/text-recognition_onnxruntime_dynamic.py

  pipeline_ort_detection_dynamic_fp32: &pipeline_ort_detection_dynamic_fp32
    convert_image: *convert_image_det
    deploy_config: configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py

tensorrt:
  pipeline_trt_recognition_dynamic_fp16: &pipeline_trt_recognition_dynamic_fp16
    convert_image: *convert_image_rec
    backend_test: *default_backend_test
    sdk_config: *sdk_recognition_dynamic
    deploy_config: configs/mmocr/text-recognition/text-recognition_tensorrt-fp16_dynamic-1x32x32-1x32x640.py

  pipeline_trt_detection_dynamic_fp16: &pipeline_trt_detection_dynamic_fp16
    convert_image: *convert_image_det
    backend_test: *default_backend_test
    sdk_config: *sdk_detection_dynamic
    deploy_config: configs/mmocr/text-detection/text-detection_tensorrt-fp16_dynamic-320x320-2240x2240.py

openvino:
  # 此处省略，内容同上
ncnn:
  # 此处省略，内容同上
pplnn:
  # 此处省略，内容同上
torchscript:
  # 此处省略，内容同上


models:
  - name: crnn # 模型名称
    metafile: configs/textrecog/crnn/metafile.yml # 模型对应的 metafile 的路径，相对于 codebase 的路径
    codebase_model_config_dir: configs/textrecog/crnn # `model_configs` 的父文件夹路径，相对于 codebase 的路径
    model_configs: # 需要测试的 config 名称
      - crnn_academic_dataset.py
    pipelines: # 使用的 pipeline
      - *pipeline_ort_recognition_dynamic_fp32

  - name: dbnet
    metafile: configs/textdet/dbnet/metafile.yml
    codebase_model_config_dir: configs/textdet/dbnet
    model_configs:
      - dbnet_r18_fpnc_1200e_icdar2015.py
    pipelines:
      - *pipeline_ort_detection_dynamic_fp32
      - *pipeline_trt_detection_dynamic_fp16

      # 特殊的 pipeline 可以这样加入
      - convert_image: xxx
        backend_test: xxx
        sdk_config: xxx
        deploy_config: configs/mmocr/text-detection/xxx
```

## 4. 生成的报告

### 模板

|      | Model    | Model Config      | Task             | Checkpoint     | Dataset    | Backend  | Deploy Config   | Static or Dynamic | Precision Type | Conversion Result | metric_1    | metric_2    | metric_n    | Test Pass    |
| ---- | -------- | ----------------- | ---------------- | -------------- | ---------- | -------- | --------------- | ----------------- | -------------- | ----------------- | ----------- | ----------- | ----------- | ------------ |
| 序号 | 模型名称 | model config 路径 | 执行的 task name | `.pth`模型路径 | 数据集名称 | 后端名称 | deploy cfg 路径 | 动态 or 静态      | 测试精度       | 模型转换结果      | 指标 1 数值 | 指标 2 数值 | 指标 n 数值 | 后端测试结果 |

### 示例

这是 MMOCR 生成的报告

|     | Model | Model Config                                                     | Task             | Checkpoint                                                                                                   | Dataset   | Backend         | Deploy Config                                                                          | Static or Dynamic | Precision Type | Conversion Result | hmean-iou | word_acc | Test Pass |
| --- | ----- | ---------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------ | --------- | --------------- | -------------------------------------------------------------------------------------- | ----------------- | -------------- | ----------------- | --------- | -------- | --------- |
| 0   | crnn  | ../mmocr/configs/textrecog/crnn/crnn_academic_dataset.py         | Text Recognition | ../mmdeploy_checkpoints/mmocr/crnn/crnn_academic-a723a1c5.pth                                                | IIIT5K    | Pytorch         | -                                                                                      | -                 | -              | -                 | -         | 80.5     | -         |
| 1   | crnn  | ../mmocr/configs/textrecog/crnn/crnn_academic_dataset.py         | Text Recognition | ${WORK_DIR}/mmocr/crnn/onnxruntime/static/crnn_academic-a723a1c5/end2end.onnx                                | x         | onnxruntime     | configs/mmocr/text-recognition/text-recognition_onnxruntime_dynamic.py                 | static            | fp32           | True              | -         | 80.67    | True      |
| 2   | crnn  | ../mmocr/configs/textrecog/crnn/crnn_academic_dataset.py         | Text Recognition | ${WORK_DIR}/mmocr/crnn/onnxruntime/static/crnn_academic-a723a1c5                                             | x         | SDK-onnxruntime | configs/mmocr/text-recognition/text-recognition_sdk_dynamic.py                         | static            | fp32           | True              | -         | x        | False     |
| 3   | dbnet | ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ../mmdeploy_checkpoints/mmocr/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth                 | ICDAR2015 | Pytorch         | -                                                                                      | -                 | -              | -                 | 0.795     | -        | -         |
| 4   | dbnet | ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ../mmdeploy_checkpoints/mmocr/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth                 | ICDAR     | onnxruntime     | configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py                     | dynamic           | fp32           | True              | -         | -        | True      |
| 5   | dbnet | ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ${WORK_DIR}/mmocr/dbnet/tensorrt/dynamic/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597/end2end.engine | ICDAR     | tensorrt        | configs/mmocr/text-detection/text-detection_tensorrt-fp16_dynamic-320x320-2240x2240.py | dynamic           | fp16           | True              | 0.793302  | -        | True      |
| 6   | dbnet | ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ${WORK_DIR}/mmocr/dbnet/tensorrt/dynamic/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597                | ICDAR     | SDK-tensorrt    | configs/mmocr/text-detection/text-detection_sdk_dynamic.py                             | dynamic           | fp16           | True              | 0.795073  | -        | True      |

## 5. 支持的后端

- [x] ONNX Runtime
- [x] TensorRT
- [x] PPLNN
- [x] ncnn
- [x] OpenVINO
- [x] TorchScript
- [x] MMDeploy SDK

## 6. 支持的Codebase及其Metric

| Codebase | Metric   | Support            |
| -------- | -------- | ------------------ |
| mmdet    | bbox     | :heavy_check_mark: |
|          | segm     | :heavy_check_mark: |
|          | PQ       | :x:                |
| mmcls    | accuracy | :heavy_check_mark: |
| mmseg    | mIoU     | :heavy_check_mark: |
| mmpose   | AR       | :heavy_check_mark: |
|          | AP       | :heavy_check_mark: |
| mmocr    | hmean    | :heavy_check_mark: |
|          | acc      | :heavy_check_mark: |
| mmedit   | PSNR     | :heavy_check_mark: |
|          | SSIM     | :heavy_check_mark: |

## 7. 注意事项

暂无

## 8. 常见问题

暂无
