# MMPose Deployment

- [MMPose Deployment](#mmpose-deployment)
  - [Installation](#installation)
    - [Install mmpose](#install-mmpose)
    - [Install mmdeploy](#install-mmdeploy)
  - [Convert model](#convert-model)
  - [Model specification](#model-specification)
  - [Model inference](#model-inference)
    - [Backend model inference](#backend-model-inference)
    - [SDK model inference](#sdk-model-inference)
  - [Supported models](#supported-models)

______________________________________________________________________

[MMPose](https://github.com/open-mmlab/mmpose/tree/main) aka `mmpose` is an open-source toolbox for pose estimation based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## Installation

### Install mmpose

Please follow the [best practice](https://mmpose.readthedocs.io/en/latest/installation.html#best-practices) to install mmpose.

### Install mmdeploy

There are several methods to install mmdeploy, among which you can choose an appropriate one according to your target platform and device.

**Method I:** Install precompiled package

You can refer to [get_started](https://mmdeploy.readthedocs.io/en/latest/get_started.html#installation)

**Method II:** Build using scripts

If your target platform is **Ubuntu 18.04 or later version**, we encourage you to run
[scripts](../01-how-to-build/build_from_script.md). For example, the following commands install mmdeploy as well as inference engine - `ONNX Runtime`.

```shell
git clone --recursive -b main https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```

**Method III:** Build from source

If neither **I** nor **II** meets your requirements, [building mmdeploy from source](../01-how-to-build/build_from_source.md) is the last option.

## Convert model

You can use [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/tree/main/tools/deploy.py) to convert mmpose models to the specified backend models. Its detailed usage can be learned from [here](https://github.com/open-mmlab/mmdeploy/tree/main/docs/en/02-how-to-run/convert_model.md#usage).

The command below shows an example about converting `hrnet` model to onnx model that can be inferred by ONNX Runtime.

```shell
cd mmdeploy
# download hrnet model from mmpose model zoo
mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 --dest .
# convert mmdet model to onnxruntime model with static shape
python tools/deploy.py \
    configs/mmpose/pose-detection_onnxruntime_static.py \
    td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    demo/resources/human-pose.jpg \
    --work-dir mmdeploy_models/mmpose/ort \
    --device cpu \
    --show
```

It is crucial to specify the correct deployment config during model conversion. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmpose) of all supported backends for mmpose. The config filename pattern is:

```
pose-detection_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml etc.
- **{precision}:** fp16, int8. When it's empty, it means fp32
- **{static | dynamic}:** static shape or dynamic shape
- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `hrnet` to other backend models by changing the deployment config file `pose-detection_onnxruntime_static.py` to [others](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmpose), e.g., converting to tensorrt model by `pose-detection_tensorrt_static-256x192.py`.

```{tip}
When converting mmpose models to tensorrt models, --device should be set to "cuda"
```

## Model specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmpose/ort` in the previous example. It includes:

```
mmdeploy_models/mmpose/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- \***.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmpose/ort** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model and visualize the results.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmpose/pose-detection_onnxruntime_static.py'
model_cfg = 'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmpose/ort/end2end.onnx']
image = './demo/resources/human-pose.jpg'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='output_pose.png')
```

### SDK model inference

TODO

## Supported models

| Model                                                                                                     | Task          | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |
| :-------------------------------------------------------------------------------------------------------- | :------------ | :----------: | :------: | :--: | :---: | :------: |
| [HRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#hrnet-cvpr-2019)          | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [MSPN](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#mspn-arxiv-2019)           | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [LiteHRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#litehrnet-cvpr-2021)  | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [Hourglass](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#hourglass-eccv-2016) | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [SimCC](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#simcc-eccv-2022)         | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)                                | PoseDetection |      Y       |    Y     |  Y   |   N   |    Y     |
| [YoloX-Pose](https://github.com/open-mmlab/mmpose/tree/main/projects/yolox_pose)                          | PoseDetection |      Y       |    Y     |  N   |   N   |    Y     |
| [RTMO](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmo)                                   | PoseDetection |      Y       |    Y     |  N   |   N   |    N     |
