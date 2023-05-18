# Model serving

MMDeploy provides model server deployment based on Triton Inference Server.

## Supported tasks

The following tasks are currently supported:

- [image-classification](../../../demo/triton/image-classification/README.md)
- [instance-segmentation](../../../demo/triton/instance-segmentation)
- [keypoint-detection](../../../demo/triton/keypoint-detection)
- [object-detection](../../../demo/triton/object-detection)
- [oriented-object-detection](../../../demo/triton/oriented-object-detection)
- [semantic-segmentation](../../../demo/triton/semantic-segmentation)
- [text-detection](../../../demo/triton/text-detection)
- [text-recognition](../../../demo/triton/text-recognition)
- [text-ocr](../../../demo/triton/text-ocr)

## Run Triton

In order to use Triton Inference Server, we need:

1. Compile MMDeploy Triton Backend
2. Prepare the model repository (including model files, and configuration files)

### Compile MMDeploy Triton Backend

a) Using Docker images

For ease of use, we provide a Docker image to support the deployment of models converted by MMDeploy. The image supports Tensorrt and ONNX Runtime as backends. If you need other backends, you can choose build from source.

b) Build from source

You can refer [build from source](../01-how-to-build/build_from_source.md) to build MMDeploy. In order to build MMDeploy Triton Backend, you need to add `-DTRITON_MMDEPLOY_BACKEND=ON` to cmake configure command. By default, the latest version of Triton Backend is used. If you want to use an older version of Triton Backend, you can add `-DTRITON_TAG=r22.12` to the cmake configure command.

### Prepare the model repository

Triton Inference Server has its own model description rules. Therefore the models converted through `tools/deploy.py ... --dump-info` need to be formatted to make Triton load correctly. We have prepared templates for each task. You can use `demo/triton/to_triton_model.py` script for model formatting. For complete samples, please refer to the description of each demo.
