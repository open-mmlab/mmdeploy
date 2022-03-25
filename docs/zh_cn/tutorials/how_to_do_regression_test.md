# 如何进行回归测试

<!-- -->
这篇教程介绍了如何进行回归测试。部署配置文件由`每个codebase的回归配置文件`， `推理框架配置信息`组成。

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
    --backend "${BACKEND}"  \
    --work-dir "${WORK_DIR}" \
    --device "${DEVICE}" \    
    --log-level INFO
```

### 参数解析

- `--deploy-yml` : 需要测试的 codebase， eg.mmdeploy/test/regression/mmdet.yml ，如果设置为 mmdeploy/test/regression/ 即全部测试。
- `--work-dir` : 模型转换、报告生成的路径。
- `--device` : 使用的设备，默认是显卡。
- `--test-type` : 测试模式：convert 测试转换，precision 测试精度。
- `--backend` : 筛选测试的后端, 默认None: 测全部backend,  也可传入若干个后端，例如 "mmdet,mmcls"。
- `--log-level` : 设置日记的等级，选项包括`'CRITICAL'， 'FATAL'， 'ERROR'， 'WARN'， 'WARNING'， 'INFO'， 'DEBUG'， 'NOTSET'`。 默认是`INFO`。

## 2. 回归测试配置文件

### 示例
```yaml
Globals:
  codebase_name: mmdet
  codebase_release: V2.22.0
  codebase_dir: ../mmdet
  checkpoint_force_download: False
  checkpoint_path: path/to/checkpoint_dir/
  metric_tolerance:
    box_AP: 0.1
    mask_AP: 0.1
  default_deploy_config:
    onnxruntime:
        static: 
          fp32: xxx.py
        dynamic: 
          fp32: xxx.py
    tensorrt:
         static: 
           fp32: xxx.py
           fp16: xxx.py
           int8: xxx.py
         dynamic:
           fp32: xxx.py
           fp16: xxx.py
           int8: xxx.py
    openvino: 
        static: 
          fp32: xxx.py
        dynamic: 
          fp32: xxx.py
    ncnn:
        static: 
          fp32: xxx.py
        dynamic: 
          fp32: xxx.py
    pplnn:
        static: 
          fp32: xxx.py
        dynamic: 
          fp32: xxx.py
    sdk: 
        static: 
           fp32: xxx.py
           fp16: xxx.py
           int8: xxx.py
        dynamic: 
           fp32: xxx.py
           fp16: xxx.py
           int8: xxx.py

Models:
  - Name: Yolo
    model_config:
      - 'configs/yolo/xxx.py'
      - 'configs/yolo/xxx.py'
    metafile: 'configs/resnet/metafile.yml'
    calib-dataset-cfg: xxx
    backends:
      - onnxruntime:
          performance_align: True
          deploy_config:
            static: 
              fp32: xxx.py
            dynamic: 
              fp32: xxx.py      
      - tensorrt:
          performance_align: True
          deploy_config:
             static: 
               fp32: xxx.py
               fp16: xxx.py
               int8: xxx.py
             dynamic:
               fp32: xxx.py
               fp16: xxx.py
               int8: xxx.py
      - ncnn:
      - openvino:
      - pplnn:
      - sdk:

  - Name: HRNet

```

### 参数解析

## 7. 注意事项

None

## 8. 常见问题

None
