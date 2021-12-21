# 如何评测模型性能

当我们把一个 PyTorch 模型转换到一个后端上后，我们或许需要在使用这个模型前对模型进行性能评测。在 MMDeploy 中，我们提供了一个模型评估工具，该工具位于 `tools/test.py`

## 准备工作

在对一个特定后端的模型进行评估前，你需要先安装该后端的插件，并且使用部署工具把模型转换到对应后端。

## 用法

```shell
python tools/test.py \
${DEPLOY_CFG} \
${MODEL_CFG} \
--model ${BACKEND_MODEL_FILES} \
[--out ${OUTPUT_PKL_FILE}] \
[--format-only] \
[--metrics ${METRICS}] \
[--show] \
[--show-dir ${OUTPUT_IMAGE_DIR}] \
[--show-score-thr ${SHOW_SCORE_THR}] \
--divice ${DEVICE} \
[--cfg-options ${CFG_OPTIONS}] \
[--metric-options $(METRIC_OPTIONS)]
```

## 参数描述

* `deploy_cfg`: 用于部署的配置文件。
* `model_cfg`: OpenMMLab 系列代码库中使用的模型配置文件。
* `--model`: 后端模型文件。比如我们把模型转换到 TensorRT 上后,我们需要在这个参数里传递一个以 ".engine" 为后缀的文件。
* `--out`:  以 pickle 格式存储的结果文件的路径. (仅在设置该参数时保存结果)
* `--format-only`: 是否仅格式化输出结果而不评测。这个参数用于把结果以特定格式生成并提交测试服务器。
* `--metrics`: 模型评测指标，定义于 OpenMMLab 各代码库中。比如 mmdet 库的COCO中的 "segm" , "proposal", mmcls 库的单标签数据集中的 "precision", "recall", "f1_score", "support"  。
* `--show`: 是否把模型结果输出到屏幕上。
* `--show-dir`: 存储结果的目录。 (仅在设置该参数时保存结果)
* `--show-score-thr`: 决定检测框是否显示的阈值.
* `--device`: 运行模型的设备。注意有些后端会限定模型运行的设备，比如 TensorRT 要求数据和模型在 cuda 上运行。
* `--cfg-options`: 附加的或覆盖的模型配置，这些配置会合入部署配置文件。
* `--metric-options`: 自定义的评测选项。选项格式以键值对 "xxx=yyy" 的形式给出，这些选项会传入 dataset.evaluate() 函数的 kwargs 参数。

\* `tools/test.py` 里还有一些用于速度测试的参数，这些参数与性能评测无关。

## 示例

```shell
python tools/test.py \
    configs/mmcls/classification_onnxruntime_static.py \
    {MMCLS_DIR}/configs/resnet/resnet50_b32x8_imagenet.py \
    --model model.onnx\
    --out out.pkl \
    --device cuda:0 \
```

## 注意事项

* OpenMMLab 各代码库模型的性能评测结果可以在本文档中各代码库的介绍页面中找到。
