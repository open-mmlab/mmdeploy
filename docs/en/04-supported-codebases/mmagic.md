# MMagic Deployment

- [MMagic Deployment](#mmagic-deployment)
  - [Installation](#installation)
    - [Install mmagic](#install-mmagic)
    - [Install mmdeploy](#install-mmdeploy)
  - [Convert model](#convert-model)
    - [Convert super resolution model](#convert-super-resolution-model)
  - [Model specification](#model-specification)
  - [Model inference](#model-inference)
    - [Backend model inference](#backend-model-inference)
    - [SDK model inference](#sdk-model-inference)
  - [Supported models](#supported-models)

______________________________________________________________________

[MMagic](https://github.com/open-mmlab/mmagic/tree/main) aka `mmagic` is an open-source image and video editing toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## Installation

### Install mmagic

Please follow the [installation guide](https://github.com/open-mmlab/mmagic/tree/main#installation) to install mmagic.

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

You can use [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/tree/main/tools/deploy.py) to convert mmagic models to the specified backend models. Its detailed usage can be learned from [here](https://github.com/open-mmlab/mmdeploy/tree/main/docs/en/02-how-to-run/convert_model.md#usage).

When using `tools/deploy.py`, it is crucial to specify the correct deployment config. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmagic) of all supported backends for mmagic, under which the config file path follows the pattern:

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{task}:** task in mmagic.

  MMDeploy supports models of one task in mmagic, i.e., `super resolution`. Please refer to chapter [supported models](#supported-models) for task-model organization.

  **DO REMEMBER TO USE** the corresponding deployment config file when trying to convert models of different tasks.

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml etc.

- **{precision}:** fp16, int8. When it's empty, it means fp32

- **{static | dynamic}:** static shape or dynamic shape

- **{shape}:** input shape or shape range of a model

### Convert super resolution model

The command below shows an example about converting `ESRGAN` model to onnx model that can be inferred by ONNX Runtime.

```shell
cd mmdeploy
# download esrgan model from mmagic model zoo
mim download mmagic --config esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k --dest .
# convert esrgan model to onnxruntime model with dynamic shape
python tools/deploy.py \
  configs/mmagic/super-resolution/super-resolution_onnxruntime_dynamic.py \
  esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py \
  esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth \
  demo/resources/face.png \
  --work-dir mmdeploy_models/mmagic/ort \
  --device cpu \
  --show \
  --dump-info
```

You can also convert the above model to other backend models by changing the deployment config file `*_onnxruntime_dynamic.py` to [others](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmagic), e.g., converting to tensorrt model by `super-resolution/super-resolution_tensorrt-_dynamic-32x32-512x512.py`.

```{tip}
When converting mmagic models to tensorrt models, --device should be set to "cuda"
```

## Model specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmagic/ort` in the previous example. It includes:

```
mmdeploy_models/mmagic/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- \***.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmagic/ort** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model and visualize the results.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmagic/super-resolution/super-resolution_onnxruntime_dynamic.py'
model_cfg = 'esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmagic/ort/end2end.onnx']
image = './demo/resources/face.png'

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
    output_file='output_restorer.bmp')
```

### SDK model inference

You can also perform SDK model inference like following,

```python
from mmdeploy_runtime import Restorer
import cv2

img = cv2.imread('./demo/resources/face.png')

# create a classifier
restorer = Restorer(model_path='./mmdeploy_models/mmagic/ort', device_name='cpu', device_id=0)
# perform inference
result = restorer(img)

# visualize inference result
# convert to BGR
result = result[..., ::-1]
cv2.imwrite('output_restorer.bmp', result)
```

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/main/demo).

## Supported models

| Model                                                                             | Task             | ONNX Runtime | TensorRT | ncnn | PPLNN | OpenVINO |
| :-------------------------------------------------------------------------------- | :--------------- | :----------: | :------: | :--: | :---: | :------: |
| [SRCNN](https://github.com/open-mmlab/mmagic/tree/main/configs/srcnn)             | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [ESRGAN](https://github.com/open-mmlab/mmagic/tree/main/configs/esrgan)           | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [ESRGAN-PSNR](https://github.com/open-mmlab/mmagic/tree/main/configs/esrgan)      | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [SRGAN](https://github.com/open-mmlab/mmagic/tree/main/configs/srgan_resnet)      | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [SRResNet](https://github.com/open-mmlab/mmagic/tree/main/configs/srgan_resnet)   | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [Real-ESRGAN](https://github.com/open-mmlab/mmagic/tree/main/configs/real_esrgan) | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
| [EDSR](https://github.com/open-mmlab/mmagic/tree/main/configs/edsr)               | super-resolution |      Y       |    Y     |  Y   |   N   |    Y     |
| [RDN](https://github.com/open-mmlab/mmagic/tree/main/configs/rdn)                 | super-resolution |      Y       |    Y     |  Y   |   Y   |    Y     |
