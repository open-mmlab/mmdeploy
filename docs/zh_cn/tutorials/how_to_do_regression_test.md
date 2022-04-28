# 如何进行回归测试

<!-- -->
这篇教程介绍了如何进行回归测试。部署配置文件由`每个codebase的回归配置文件`，`推理框架配置信息`组成。

<!-- TOC -->

- [如何进行回归测试](#如何进行回归测试)
    - [1. 用法](#1-用法)
        - [参数解析](#参数解析)
        - [示例](#示例)
    - [2. 回归测试配置文件](#2-回归测试配置文件)
        - [示例及参数解析](#示例及参数解析)
    - [3. 生成的报告](#3-生成的报告)
        - [模板](#模板)
        - [示例](#示例)
    - [4. 支持的后端](#4-支持的后端)
    - [5. 支持的Codebase及其Metric](#5-支持的Codebase及其Metric)
    - [6. 注意事项](#7-注意事项)
    - [7. 常见问题](#8-常见问题)

<!-- TOC -->

## 1. 用法

```shell
python ./tools/regression_test.py \
    --deploy-yml "${DEPLOY_YML_PATH}" \
    --backends "${BACKEND}" \
    --work-dir "${WORK_DIR}" \
    --device "${DEVICE}" \
    --log-level INFO \
    [--performance]
```

### 参数解析

- `--deploy-yml` : 需要测试的 codebase，eg.`configs/mmdet/mmdet_regression_test.yaml`，如果设置为 `all` 即全部测试。
- `--backends` : 筛选测试的后端, 默认 `all`: 测全部`backend`, 也可传入若干个后端，例如 `onnxruntime tesnsorrt`。
- `--work-dir` : 模型转换、报告生成的路径。
- `--device` : 使用的设备，默认 `cuda`。
- `--log-level` : 设置日记的等级，选项包括`'CRITICAL'， 'FATAL'， 'ERROR'， 'WARN'， 'WARNING'， 'INFO'， 'DEBUG'， 'NOTSET'`。默认是`INFO`。
- `--performance` : 是否测试精度，加上则测试转换+精度，不加上则只测试转换

### 注意事项
对于 Windows 用户：
1. 要在 shell 命令中使用 `&&` 连接符，需要下载并使用 `PowerShell 7 Preview 5+`。
2. 如果您使用 conda env，可能需要在 regression_test.py 中将 `python3` 更改为 `python`，因为 `%USERPROFILE%\AppData\Local\Microsoft\WindowsApps` 目录中有 `python3.exe`。

## 例子

1. 测试 mmdet 和 mmpose 的所有 backend 的 转换+精度

```shell
python ./tools/regression_test.py \
    --deploy-yml ./configs/mmdet/mmdet_regression_test.yaml ./configs/mmpose/mmpose_regression_test.yaml \
    --backends all \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO \
    --performance
```

2. 测试 mmdet 和 mmpose 的某几个 backend 的 转换+精度

```shell
python ./tools/regression_test.py \
    --deploy-yml ./configs/mmdet/mmdet_regression_test.yaml ./configs/mmdet/mmpose.yaml \
    --backends onnxruntime tesnsorrt \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO \
    --performance
```

3. 测试 mmdet 和 mmpose 的某几个 backend，只需测试转换

```shell
python ./tools/regression_test.py \
    --deploy-yml ./configs/mmdet/mmdet_regression_test.yaml ./configs/mmdet/mmpose.yaml \
    --backends onnxruntime tesnsorrt \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO
```

## 2. 回归测试配置文件

### 示例及参数解析

```yaml
globals:
  codebase_name: mmocr # 回归测试的 codebase 名称
  codebase_dir: ../mmocr # 回归测试的 codebase 路径
  checkpoint_force_download: False # 回归测试是否重新下载模型即使其已经存在
  checkpoint_dir: ../mmdeploy_checkpoints # 回归测试是否下载模型的路径
  images: # 测试使用图片
    img_224x224: &img_224x224 ./tests/data/tiger.jpeg
    img_300x300: &img_300x300
    img_800x1344: &img_cityscapes_800x1344
    img_blank: &img_blank
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
  convert_image: &convert_image # 转换会使用到的图片
    input_img: *img_224x224
    test_img: *img_300x300
  backend_test: &default_backend_test True # 是否对 backend 进行精度测试
  sdk: # SDK 配置文件
    sdk_detection_dynamic: &sdk_detection_dynamic configs/mmocr/text-detection/text-detection_sdk_dynamic.py
    sdk_recognition_dynamic: &sdk_recognition_dynamic configs/mmocr/text-recognition/text-recognition_sdk_dynamic.py

onnxruntime:
  pipeline_ort_recognition_static_fp32: &pipeline_ort_recognition_static_fp32
    convert_image: *convert_image # 转换过程中使用的图片
    backend_test: *default_backend_test # 是否进行后端测试，存在则判断，不存在则视为 False
    sdk_config: *sdk_recognition_dynamic # 是否进行SDK测试，存在则使用特定的 SDK config 进行测试，不存在则视为不进行 SDK 测试
    deploy_config: configs/mmocr/text-recognition/text-recognition_onnxruntime_static.py # 使用的 deploy cfg 路径，基于 mmdeploy 的路径

  pipeline_ort_recognition_dynamic_fp32: &pipeline_ort_recognition_dynamic_fp32
    convert_image: *convert_image
    backend_test: *default_backend_test
    sdk_config: *sdk_recognition_dynamic
    deploy_config: configs/mmocr/text-recognition/text-recognition_onnxruntime_dynamic.py

  pipeline_ort_detection_dynamic_fp32: &pipeline_ort_detection_dynamic_fp32
    convert_image: *convert_image
    deploy_config: configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py

tensorrt:
  pipeline_trt_recognition_dynamic_fp16: &pipeline_trt_recognition_dynamic_fp16
    convert_image: *convert_image
    backend_test: *default_backend_test
    sdk_config: *sdk_recognition_dynamic
    deploy_config: configs/mmocr/text-recognition/text-recognition_tensorrt-fp16_dynamic-1x32x32-1x32x640.py

  pipeline_trt_detection_dynamic_fp16: &pipeline_trt_detection_dynamic_fp16
    convert_image: *convert_image
    backend_test: *default_backend_test
    sdk_config: *sdk_detection_dynamic
    deploy_config: configs/mmocr/text-detection/text-detection_tensorrt-fp16_dynamic-320x320-1024x1824.py

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
```

## 3. 生成的报告

### 模板

|| model_name | model_config | task_name       | model_checkpoint_name | dataset  | backend_name | deploy_config | static_or_dynamic | precision_type | conversion_result | fps | metric_1 | metric_2 | metric_n | test_pass |
|------------|--------------|-----------------|-----------------------|----------|--------------|---------------|-------------------|----------------|-------------------|---|----------|----------|-----------|-----------|-----|
| 序号         | 模型名称         | model config 路径 | 执行的 task name      | `.pth`模型路径 | 数据集名称        | 后端名称    |  deploy cfg 路径    | 动态 or 静态          | 测试精度           | 模型转换结果       | FPS 数值     | 指标 1 数值 | 指标 2 数值        | 指标 n 数值         |  后端测试结果  |

### 示例

这是 MMOCR 生成的报告

|| model_name | model_config    | task_name  | model_checkpoint_name    | dataset   | backend_name    | deploy_config   | static_or_dynamic | precision_type | conversion_result | fps    | hmean-iou | word_acc | test_pass |
| ---- | ---------- | ------------------------------------------------------------ | ---------------- | ------------------------------------------------------------ | --------- | --------------- | ------------------------------------------------------------ | ----------------- | -------------- | ----------------- |-----------|----------|-----------| --------- |
| 0    | crnn | ../mmocr/configs/textrecog/crnn/crnn_academic_dataset.py     | Text Recognition | ../mmdeploy_checkpoints/mmocr/crnn/crnn_academic-a723a1c5.pth | IIIT5K    | Pytorch| -| -  | -     | -  | -         | -        | 80.5      | -|
| 1    | crnn | ../mmocr/configs/textrecog/crnn/crnn_academic_dataset.py     | Text Recognition | ${WORK_DIR}/mmocr/crnn/onnxruntime/static/crnn_academic-a723a1c5/end2end.onnx | x| onnxruntime     | configs/mmocr/text-recognition/text-recognition_onnxruntime_dynamic.py | static   | fp32  | True     | 182.21    | -        | 80.67     | True|
| 2    | crnn | ../mmocr/configs/textrecog/crnn/crnn_academic_dataset.py     | Text Recognition | ${WORK_DIR}/mmocr/crnn/onnxruntime/static/crnn_academic-a723a1c5 | x| SDK-onnxruntime | configs/mmocr/text-recognition/text-recognition_sdk_dynamic.py | static   | fp32  | True     | x         | -        | x         | False     |
| 3    | dbnet| ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ../mmdeploy_checkpoints/mmocr/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth | ICDAR2015 | Pytorch| -| -  | -     | -  | -         | 0.795    | -         | -|
| 4    | dbnet| ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ../mmdeploy_checkpoints/mmocr/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth | ICDAR     | onnxruntime     | configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py | dynamic  | fp32  | True     | -         | -        | -         | True|
| 5    | dbnet| ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ${WORK_DIR}/mmocr/dbnet/tensorrt/dynamic/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597/end2end.engine | ICDAR     | tensorrt  | configs/mmocr/text-detection/text-detection_tensorrt-fp16_dynamic-320x320-1024x1824.py | dynamic  | fp16  | True     | 229.06    | 0.793302 | -  | True|
| 6    | dbnet| ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ${WORK_DIR}/mmocr/dbnet/tensorrt/dynamic/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597 | ICDAR     | SDK-tensorrt    | configs/mmocr/text-detection/text-detection_sdk_dynamic.py   | dynamic  | fp16  | True     | 140.06    | 0.795073 | -         | True|

## 4. 支持的后端
- [x] ONNX Runtime
- [x] TensorRT
- [x] PPLNN
- [x] ncnn
- [x] OpenVINO
- [x] TorchScript
- [x] MMDeploy SDK

## 5. 支持的Codebase及其Metric
- [x] mmdet
  - [x] bbox
- [x] mmcls
  - [x] accuracy
- [x] mmseg
  - [x] mIoU
- [x] mmpose
  - [x] AR
  - [x] AP
- [x] mmocr
  - [x] hmean
  - [x] acc
- [x] mmedit
  - [x] PSNR
  - [x] SSIM

## 6. 注意事项

暂无

## 7. 常见问题

暂无
