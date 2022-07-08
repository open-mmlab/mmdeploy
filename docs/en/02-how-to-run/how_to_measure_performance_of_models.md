# How to profile model

After converting a PyTorch model to a backend model, you can profile inference speed using `tools/test.py`.

## Prerequisite

Install MMDeploy according to [get-started](../get_started.md) instructions.
And convert the PyTorch model or ONNX model to the backend model by following the [guide](convert_model.md).

## Profile

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

## Description of all arguments

- `deploy_cfg`: The config for deployment.
- `model_cfg`: The config of the model in OpenMMLab codebases.
- `--model`: The backend model files. For example, if we convert a model to ncnn, we need to pass a ".param" file and a ".bin" file. If we convert a model to TensorRT, we need to pass the model file with ".engine" suffix.
- `--log2file`: log evaluation results and speed to file.
- `--speed-test`:  Whether to activate speed test.
- `--warmup`: warmup before counting inference elapse, require setting speed-test first.
- `--log-interval`: The interval between each log, require setting speed-test first.

\* Other arguments in `tools/test.py` are used for performance test. They have no concern with speed test.

## Example

```shell
python tools/test.py \
    configs/mmcls/classification_onnxruntime_static.py \
    {MMCLS_DIR}/configs/resnet/resnet50_b32x8_imagenet.py \
    --model model.onnx \
    --speed-test \
    --device cpu
```
