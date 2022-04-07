# 如何进行回归测试

<!-- -->
这篇教程介绍了如何进行回归测试。部署配置文件由`每个codebase的回归配置文件`，`推理框架配置信息`组成。

<!-- TOC -->

- [如何进行回归测试](#如何进行回归测试)
    - [1. 用法](#1-用法)
        - [示例](#示例)
        - [参数解析](#示例)
    - [2. 回归测试配置文件](#2-回归测试配置文件)
        - [示例](#示例)
        - [参数解析](#示例)
    - [7. 注意事项](#7-注意事项)
    - [8. 常见问题](#8-常见问题)

<!-- TOC -->

## 1. 用法

```shell
python ./tools/regression_test.py \
    --deploy-yml "${DEPLOY_YML_PATH}" \
    --test-type "${TEST_TYPE}" \
    --backend "${BACKEND}" \
    --work-dir "${WORK_DIR}" \
    --device "${DEVICE}" \
    --log-level INFO
```

### 参数解析

- `--deploy-yml` : 需要测试的 codebase，eg.mmdeploy/test/regression/mmdet.yml，如果设置为 mmdeploy/test/regression/ 即全部测试。
- `--work-dir` : 模型转换、报告生成的路径。
- `--device` : 使用的设备，默认是显卡。
- `--test-type` : 测试模式：convert 测试转换，precision 测试精度。
- `--backend` : 筛选测试的后端, 默认None: 测全部backend, 也可传入若干个后端，例如 "mmdet,mmcls"。
- `--log-level` : 设置日记的等级，选项包括`'CRITICAL'， 'FATAL'， 'ERROR'， 'WARN'， 'WARNING'， 'INFO'， 'DEBUG'， 'NOTSET'`。默认是`INFO`。

## 例子

1. 测试 mmdet 和 mmpose 的所有 backend

```shell
python ./tools/regression_test.py \
    --deploy-yml ./configs/mmdet/mmdet_regression_test.yaml ./configs/mmdet/mmpose.yaml \
    --test-type "precision" \
    --backends all \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO
```

2. 测试 mmdet 和 mmpose 的某几个 backend

```shell
python ./tools/regression_test.py \
    --deploy-yml ./configs/mmdet/mmdet_regression_test.yaml ./configs/mmdet/mmpose.yaml \
    --test-type "precision" \
    --backends onnxruntime tesnsorrt \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO
```

## 2. 回归测试配置文件

### 示例

```yaml
globals:
  codebase_name: mmdet
  codebase_dir: ../mmdetection
  checkpoint_force_download: False
  checkpoint_dir: ../mmdeploy_checkpoints
  images:
    img_coco_320x320: &img_coco_320x320 ./tests/data/tiger.jpeg
    img_coco_300x300: &img_coco_300x300
    img_coco_800x1344: &img_coco_800x1344
    img_blank: &img_blank
  dataset_path:
    COCO:
      image:
      annotation:
  metric_info: &metric_info
    box AP:
      eval_name: bbox       #  test.py --metrics args
      metric_key: bbox_mAP  #  eval OrderedDict key name
      tolerance: 0.1
    mask AP:
      eval_name: segm
      metric_key: '?'
      tolerance: 0.1
    PQ:
      eval_name: proposal
      metric_key: '?'
      tolerance: 0.1
  convert_image: &convert_image
    input_img: *img_coco_320x320
    test_img: *img_coco_300x300
  backend_test: &default_backend_test True
  sdk:
    sdk_dynamic_fp32: &sdk_dynamic_fp32 configs/mmdet/detection/detection_sdk_dynamic.py

onnxruntime:
  pipeline_ort_static_fp32: &pipeline_ort_static_fp32
    <<: *metric_info
    <<: *convert_image
    deploy_config: configs/mmdet/detection/detection_onnxruntime_static.py

  pipeline_ort_dynamic_fp32: &pipeline_ort_dynamic_fp32
    <<: *metric_info
    <<: *convert_image
    deploy_config: configs/mmdet/detection/detection_onnxruntime_dynamic.py

tensorrt:
  pipeline_trt_static_fp32: &pipeline_trt_static_fp32
    <<: *metric_info
    <<: *convert_image
    backend_test: *default_backend_test
    deploy_config: configs/mmdet/detection/detection_tensorrt_static-800x1344.py

  pipeline_trt_dynamic_fp32: &pipeline_trt_dynamic_fp32
    <<: *metric_info
    <<: *convert_image
    backend_test: *default_backend_test
    sdk_config: *sdk_dynamic_fp32
    deploy_config: configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py

  pipeline_trt_dynamic_fp16: &pipeline_trt_dynamic_fp16
    <<: *metric_info
    <<: *convert_image
    backend_test: *default_backend_test
    sdk_config: *sdk_dynamic_fp32
    deploy_config: configs/mmdet/detection/detection_tensorrt-fp16_dynamic-320x320-1344x1344.py

  pipeline_trt_dynamic_int8: &pipeline_trt_dynamic_int8
    <<: *metric_info
    <<: *convert_image
    backend_test: *default_backend_test
    sdk_config: *sdk_dynamic_fp32
    deploy_config: configs/mmdet/detection/detection_tensorrt-int8_dynamic-300x300-512x512.py

openvino:
ncnn:
pplnn:
torchscript:


models:
  - name: retinanet
    codebase_model_config_dir: ./configs/retinanet
    metafile: configs/retinanet/metafile.yml
    model_configs:
      - retinanet_r50_fpn_1x_coco.py
    pipelines:
      - *pipeline_ort_dynamic_fp32
      - *pipeline_trt_dynamic_fp16
```

### 参数解析

## 7. 注意事项

None

## 8. 常见问题

None
