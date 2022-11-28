# How to do regression test

This tutorial describes how to do regression test. The deployment configuration file contains  codebase config and inference config.

### 1. Python Environment

```shell
pip install -r requirements/tests.txt
```

If pip throw an exception, try to upgrade numpy.

```shell
pip install -U numpy
```

## 2. Usage

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

### Description

- `--codebase` : The codebase to test, eg.`mmdet`. If you want to test multiple codebase, use `mmcls mmdet ...`
- `--backends` : The backend to test. By default, all `backend`s would be tested. You can use `onnxruntime tesensorrt`to choose several backends. If you also need to test the SDK, you need to configure the `sdk_config` in `tests/regression/${codebase}.yml`.
- `--models` : Specify the model to be tested. All models in `yml` are tested by default. You can also give some model names. For the model name, please refer to the relevant yml configuration file. For example `ResNet SE-ResNet "Mask R-CNN"`. Model name can only contain numbers and letters.
- `--work-dir` : The directory of model convert and report, use `../mmdeploy_regression_working_dir` by default.
- `--checkpoint-dir`: The path of downloaded torch model, use `../mmdeploy_checkpoints` by default.
- `--device` : device type, use `cuda` by default
- `--log-level` : These options are available:`'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG',  'NOTSET'`. The default value is `INFO`.
- `-p` or `--performance` : Test precision or not. If not enabled, only model convert would be tested.

### Notes

For Windows user:

1. To use the `&&` connector in shell commands, you need to download `PowerShell 7 Preview 5+`.
2. If you are using conda env, you may need to change `python3` to `python` in regression_test.py because there is `python3.exe` in `%USERPROFILE%\AppData\Local\Microsoft\WindowsApps` directory.

## Example

1. Test all backends of mmdet and mmpose for **model convert and precision**

```shell
python ./tools/regression_test.py \
    --codebase mmdet mmpose \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO \
    --performance
```

2. Test **model convert and precision** of some backends of mmdet and mmpose

```shell
python ./tools/regression_test.py \
    --codebase mmdet mmpose \
    --backends onnxruntime tensorrt \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO \
    -p
```

3. Test some backends of mmdet and mmpose, **only test model convert**

```shell
python ./tools/regression_test.py \
    --codebase mmdet mmpose \
    --backends onnxruntime tensorrt \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO
```

4. Test some models of mmdet and mmcls, **only test model convert**

```shell
python ./tools/regression_test.py \
    --codebase mmdet mmpose \
    --models ResNet SE-ResNet "Mask R-CNN" \
    --work-dir "../mmdeploy_regression_working_dir" \
    --device "cuda" \
    --log-level INFO
```

## 3. Regression Test Tonfiguration

### Example and parameter description

```yaml
globals:
  codebase_dir: ../mmocr # codebase path to test
  checkpoint_force_download: False # whether to redownload the model even if it already exists
  images:
    img_densetext_det: &img_densetext_det ../mmocr/demo/demo_densetext_det.jpg
    img_demo_text_det: &img_demo_text_det ../mmocr/demo/demo_text_det.jpg
    img_demo_text_ocr: &img_demo_text_ocr ../mmocr/demo/demo_text_ocr.jpg
    img_demo_text_recog: &img_demo_text_recog ../mmocr/demo/demo_text_recog.jpg
  metric_info: &metric_info
    hmean-iou: # metafile.Results.Metrics
      eval_name: hmean-iou #  test.py --metrics args
      metric_key: 0_hmean-iou:hmean # the key name of eval log
      tolerance: 0.1 # tolerated threshold interval
      task_name: Text Detection # the name of metafile.Results.Task
      dataset: ICDAR2015 # the name of metafile.Results.Dataset
    word_acc: # same as hmean-iou, also a kind of metric
      eval_name: acc
      metric_key: 0_word_acc_ignore_case
      tolerance: 0.2
      task_name: Text Recognition
      dataset: IIIT5K
  convert_image_det: &convert_image_det # the image that will be used by detection model convert
    input_img: *img_densetext_det
    test_img: *img_demo_text_det
  convert_image_rec: &convert_image_rec
    input_img: *img_demo_text_recog
    test_img: *img_demo_text_recog
  backend_test: &default_backend_test True # whether test model precision for backend
  sdk: # SDK config
    sdk_detection_dynamic: &sdk_detection_dynamic configs/mmocr/text-detection/text-detection_sdk_dynamic.py
    sdk_recognition_dynamic: &sdk_recognition_dynamic configs/mmocr/text-recognition/text-recognition_sdk_dynamic.py

onnxruntime:
  pipeline_ort_recognition_static_fp32: &pipeline_ort_recognition_static_fp32
    convert_image: *convert_image_rec # the image used by model conversion
    backend_test: *default_backend_test # whether inference on the backend
    sdk_config: *sdk_recognition_dynamic # test SDK or not. If it exists, use a specific SDK config for testing
    deploy_config: configs/mmocr/text-recognition/text-recognition_onnxruntime_static.py # the deploy cfg path to use, based on mmdeploy path

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
  # same as onnxruntime backend configuration
ncnn:
  # same as onnxruntime backend configuration
pplnn:
  # same as onnxruntime backend configuration
torchscript:
  # same as onnxruntime backend configuration


models:
  - name: crnn # model name
    metafile: configs/textrecog/crnn/metafile.yml # the path of model metafile, based on codebase path
    codebase_model_config_dir: configs/textrecog/crnn # the basepath of `model_configs`, based on codebase path
    model_configs: # the config name to teset
      - crnn_academic_dataset.py
    pipelines: # pipeline name
      - *pipeline_ort_recognition_dynamic_fp32

  - name: dbnet
    metafile: configs/textdet/dbnet/metafile.yml
    codebase_model_config_dir: configs/textdet/dbnet
    model_configs:
      - dbnet_r18_fpnc_1200e_icdar2015.py
    pipelines:
      - *pipeline_ort_detection_dynamic_fp32
      - *pipeline_trt_detection_dynamic_fp16

      # special pipeline can be added like this
      - convert_image: xxx
        backend_test: xxx
        sdk_config: xxx
        deploy_config: configs/mmocr/text-detection/xxx
```

## 4. Generated Report

This is an example of mmocr regression test report.

|     | Model | Model Config                                                     | Task             | Checkpoint                                                                                                   | Dataset   | Backend         | Deploy Config                                                                          | Static or Dynamic | Precision Type | Conversion Result | hmean-iou | word_acc | Test Pass |
| --- | ----- | ---------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------ | --------- | --------------- | -------------------------------------------------------------------------------------- | ----------------- | -------------- | ----------------- | --------- | -------- | --------- |
| 0   | crnn  | ../mmocr/configs/textrecog/crnn/crnn_academic_dataset.py         | Text Recognition | ../mmdeploy_checkpoints/mmocr/crnn/crnn_academic-a723a1c5.pth                                                | IIIT5K    | Pytorch         | -                                                                                      | -                 | -              | -                 | -         | 80.5     | -         |
| 1   | crnn  | ../mmocr/configs/textrecog/crnn/crnn_academic_dataset.py         | Text Recognition | ${WORK_DIR}/mmocr/crnn/onnxruntime/static/crnn_academic-a723a1c5/end2end.onnx                                | x         | onnxruntime     | configs/mmocr/text-recognition/text-recognition_onnxruntime_dynamic.py                 | static            | fp32           | True              | -         | 80.67    | True      |
| 2   | crnn  | ../mmocr/configs/textrecog/crnn/crnn_academic_dataset.py         | Text Recognition | ${WORK_DIR}/mmocr/crnn/onnxruntime/static/crnn_academic-a723a1c5                                             | x         | SDK-onnxruntime | configs/mmocr/text-recognition/text-recognition_sdk_dynamic.py                         | static            | fp32           | True              | -         | x        | False     |
| 3   | dbnet | ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ../mmdeploy_checkpoints/mmocr/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth                 | ICDAR2015 | Pytorch         | -                                                                                      | -                 | -              | -                 | 0.795     | -        | -         |
| 4   | dbnet | ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ../mmdeploy_checkpoints/mmocr/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth                 | ICDAR     | onnxruntime     | configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py                     | dynamic           | fp32           | True              | -         | -        | True      |
| 5   | dbnet | ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ${WORK_DIR}/mmocr/dbnet/tensorrt/dynamic/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597/end2end.engine | ICDAR     | tensorrt        | configs/mmocr/text-detection/text-detection_tensorrt-fp16_dynamic-320x320-2240x2240.py | dynamic           | fp16           | True              | 0.793302  | -        | True      |
| 6   | dbnet | ../mmocr/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py | Text Detection   | ${WORK_DIR}/mmocr/dbnet/tensorrt/dynamic/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597                | ICDAR     | SDK-tensorrt    | configs/mmocr/text-detection/text-detection_sdk_dynamic.py                             | dynamic           | fp16           | True              | 0.795073  | -        | True      |

## 5. Supported Backends

- [x] ONNX Runtime
- [x] TensorRT
- [x] PPLNN
- [x] ncnn
- [x] OpenVINO
- [x] TorchScript
- [x] SNPE
- [x] MMDeploy SDK

## 6. Supported Codebase and Metrics

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
