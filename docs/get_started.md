## Get Started

MMDeploy provides some useful tools. It is easy to deploy models in OpenMMLab to various platforms. You can convert models in our pre-defined pipeline or build a custom conversion pipeline by yourself. This guide will show you how to convert a model with MMDeploy!

### Prerequisites

First we should install MMDeploy following [build.md](./build.md). Note that the build steps are slightly different among the supported backends. Here are some brief introductions to these backends:

- [ONNXRuntime](./backends/onnxruntime.md): ONNX Runtime is a cross-platform inference and training machine-learning accelerator. It has best support for <span style="color:red">ONNX IR</span>.
- [TensorRT](./backends/tensorrt.md): NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning inference applications. It is a good choice if you want to deploy your model on <span style="color:red">NVIDIA devices</span>.
- [ncnn](./backends/ncnn.md): ncnn is a high-performance neural network inference computing framework optimized for <span style="color:red">mobile platforms</span>. ncnn is deeply considerate about deployment and uses on <span style="color:red">mobile phones</span> from the beginning of design.
- [PPL](./backends/ppl.md): PPLNN, which is short for "PPLNN is a Primitive Library for Neural Network", is a high-performance deep-learning inference engine for efficient AI inferencing. It can run various ONNX models and has <span style="color:red">better support for OpenMMLab</span>.
- [OpenVINO](./backends/openvino.md): OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference. The open-source toolkit allows to seamlessly integrate with <span style="color:red">Intel AI hardware</span>, the latest neural network accelerator chips, the Intel AI stick, and embedded computers or edge devices.

Choose the backend which can meet your demand and install it following the link provided above.

### Convert Model

Once you have installed MMDeploy, you can convert the PyTorch model in the OpenMMLab model zoo to the backend model with one magic spell! For example, if you want to convert the Faster-RCNN in [MMDetection](https://github.com/open-mmlab/mmdetection) to TensorRT:

```bash
# Assume you have installed MMDeploy in ${MMDEPLOY_DIR} and MMDetection in ${MMDET_DIR}
# If you do not know where to find the path. Just type `pip show mmdeploy` and `pip show mmdet` in your console.

python ${MMDEPLOY_DIR}/tools/deploy.py \
    ${MMDEPLOY_DIR}/configs/mmdet/two-stage_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    ${CHECKPOINT_DIR}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ${INPUT_IMG} \
    --work-dir ${WORK_DIR} \
    --device cuda:0
```

`${MMDEPLOY_DIR}/tools/deploy.py` is a tool that does everything you need to convert a model. Read [how_to_convert_model](./tutorials/how_to_convert_model.md) for more details. The converted model and other meta-info will be found in `${WORK_DIR}`.

`two-stage_tensorrt_dynamic-320x320-1344x1344.py` is a config file that contains all arguments you need to customize the conversion pipeline. The name is formed as

```bash
<task name>_<backend>_[backend options]_<dynamic support>.py
```

It is easy to find the deployment config you need by name. If you want to customize the conversion, you can edit the config file by yourself. Here is a tutorial about [how to write config](./tutorials/how_to_write_config.md).

### Inference Model

Now you can do model inference with the APIs provided by the backend. But what if you want to test the model instantly? We have some backend wrappers for you.

```python
from mmdeploy.apis import inference_model

result = inference_model(model_cfg, deploy_cfg, backend_models, img=img, device=device)
```

The `inference_model` will create a wrapper module and do the inference for you. The result has the same format as the original OpenMMLab repo.

### Evaluate Model

You might wonder that does the backend model have the same precision as the original one? How fast can the model run? MMDeploy provides tools to test the model. Take the converted TensorRT Faster-RCNN as an example:

```bash
python ${MMDEPLOY_DIR}/tools/test.py \
    ${MMDEPLOY_DIR}/configs/mmdet/two-stage_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --model ${BACKEND_MODEL_FILES} \
    --metrics ${METRICS} \
    --device cuda:0
```

Read [how to evaluate a model](./tutorials/how_to_evaluate_a_model.md) for more details about how to use `tools/test.py`

### Add New Model Support?

If the model you want to deploy has not been supported yet in MMDeploy, you can try to support it with the `rewriter` by yourself. Rewriting the functions with control flow or unsupported ops is a good way to solve the problem.

```python
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.repeat', backend='tensorrt')
def repeat_static(ctx, input, *size):
    origin_func = ctx.origin_func
    if input.dim() == 1 and len(size) == 1:
        return origin_func(input.unsqueeze(0), *([1] + list(size))).squeeze(0)
    else:
        return origin_func(input, *size)
```

Read [how_to_support_new_models](./tutorials/how_to_support_new_models.md) to learn more about the rewriter. And, PR is welcome!
