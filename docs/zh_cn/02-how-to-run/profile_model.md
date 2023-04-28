# 如何 Profile 模型

模型转换结束后，MMDeploy 提供了 `tools/test.py` 做为单测工具。

## 依赖

需要参照 [安装说明](../01-how-to-build/build_from_source.md) 完成依赖安装
，按照 [转换说明](../02-how-to-run/convert_model.md) 转出模型。

## 用法

```shell
python tools/test.py \
${DEPLOY_CFG} \
${MODEL_CFG} \
--model ${BACKEND_MODEL_FILES} \
[--speed-test] \
[--warmup ${WARM_UP}] \
[--log-interval ${LOG_INTERVERL}] \
[--log2file ${LOG_RESULT_TO_FILE}]
```

## 参数详解

| 参数         | 说明                      |
| ------------ | ------------------------- |
| deploy_cfg   | 部署配置文件              |
| model_cfg    | codebase 中的模型配置文件 |
| log2file     | 保存日志和运行文件的路径  |
| speed-test   | 是否做速度测试            |
| warm-up      | 执行前是否 warm-up        |
| log-interval | 日志打印间隔              |

## 使用样例

执行模型推理

```shell
python tools/test.py \
    configs/mmpretrain/classification_onnxruntime_static.py \
    {MMPRETRAIN_DIR}/configs/resnet/resnet50_b32x8_imagenet.py \
    --model model.onnx \
    --out out.pkl \
    --device cuda:0
```

profile 速度测试

```shell
python tools/test.py \
    configs/mmpretrain/classification_onnxruntime_static.py \
    {MMPRETRAIN_DIR}/configs/resnet/resnet50_b32x8_imagenet.py \
    --model model.onnx \
    --speed-test \
    --device cpu
```
