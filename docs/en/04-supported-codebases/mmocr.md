# MMOCR Deployment

- [MMOCR Deployment](#mmocr-deployment)
  - [Installation](#installation)
    - [Install mmocr](#install-mmocr)
    - [Install mmdeploy](#install-mmdeploy)
  - [Convert model](#convert-model)
    - [Convert text detection model](#convert-text-detection-model)
    - [Convert text recognition model](#convert-text-recognition-model)
  - [Model specification](#model-specification)
  - [Model Inference](#model-inference)
    - [Backend model inference](#backend-model-inference)
    - [SDK model inference](#sdk-model-inference)
      - [Text detection SDK model inference](#text-detection-sdk-model-inference)
      - [Text Recognition SDK model inference](#text-recognition-sdk-model-inference)
  - [Supported models](#supported-models)
  - [Reminder](#reminder)

______________________________________________________________________

[MMOCR](https://github.com/open-mmlab/mmocr/tree/main) aka `mmocr` is an open-source toolbox based on PyTorch and mmdetection for text detection, text recognition, and the corresponding downstream tasks including key information extraction. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## Installation

### Install mmocr

Please follow the [installation guide](https://mmocr.readthedocs.io/en/latest/get_started/install.html) to install mmocr.

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

You can use [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/tree/main/tools/deploy.py) to convert mmocr models to the specified backend models. Its detailed usage can be learned from [here](https://github.com/open-mmlab/mmdeploy/tree/main/docs/en/02-how-to-run/convert_model.md#usage).

When using `tools/deploy.py`, it is crucial to specify the correct deployment config. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmocr) of all supported backends for mmocr, under which the config file path follows the pattern:

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{task}:** task in mmocr.

  MMDeploy supports models of two tasks of mmocr, one is `text detection` and the other is `text-recogntion`.

  **DO REMEMBER TO USE** the corresponding deployment config file when trying to convert models of different tasks.

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml etc.

- **{precision}:** fp16, int8. When it's empty, it means fp32

- **{static | dynamic}:** static shape or dynamic shape

- **{shape}:** input shape or shape range of a model

In the next two chapters, we will task `dbnet` model from `text detection` task and `crnn` model from `text recognition` task respectively as examples, showing how to convert them to onnx model that can be inferred by ONNX Runtime.

### Convert text detection model

```shell
cd mmdeploy
# download dbnet model from mmocr model zoo
mim download mmocr --config dbnet_resnet18_fpnc_1200e_icdar2015 --dest .
# convert mmocr model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py \
    dbnet_resnet18_fpnc_1200e_icdar2015.py \
    dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth \
    demo/resources/text_det.jpg \
    --work-dir mmdeploy_models/mmocr/dbnet/ort \
    --device cpu \
    --show \
    --dump-info
```

### Convert text recognition model

```shell
cd mmdeploy
# download crnn model from mmocr model zoo
mim download mmocr --config crnn_mini-vgg_5e_mj --dest .
# convert mmocr model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmocr/text-recognition/text-recognition_onnxruntime_dynamic.py \
    crnn_mini-vgg_5e_mj.py \
    crnn_mini-vgg_5e_mj_20220826_224120-8afbedbb.pth \
    demo/resources/text_recog.jpg \
    --work-dir mmdeploy_models/mmocr/crnn/ort \
    --device cpu \
    --show \
    --dump-info
```

You can also convert the above models to other backend models by changing the deployment config file `*_onnxruntime_dynamic.py` to [others](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmocr), e.g., converting `dbnet` to tensorrt-fp32 model by `text-detection/text-detection_tensorrt-_dynamic-320x320-2240x2240.py`.

```{tip}
When converting mmocr models to tensorrt models, --device should be set to "cuda"
```

## Model specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmocr/dbnet/ort` in the previous example. It includes:

```
mmdeploy_models/mmocr/dbnet/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- \***.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmocr/dbnet/ort** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model Inference

### Backend model inference

Take the previous converted `end2end.onnx` mode of `dbnet` as an example, you can use the following code to inference the model and visualize the results.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmocr/text-detection/text-detection_onnxruntime_dynamic.py'
model_cfg = 'dbnet_resnet18_fpnc_1200e_icdar2015.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmocr/dbnet/ort/end2end.onnx']
image = './demo/resources/text_det.jpg'

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
    output_file='output_ocr.png')
```

**Tip**:

Map 'deploy_cfg', 'model_cfg', 'backend_model' and 'image' to corresponding arguments in chapter [convert text recognition model](#convert-text-recognition-model), you will get the ONNX Runtime inference results of `crnn` onnx model.

### SDK model inference

Given the above SDK models of `dbnet` and `crnn`, you can also perform SDK model inference like following,

#### Text detection SDK model inference

```python
import cv2
from mmdeploy_runtime import TextDetector

img = cv2.imread('demo/resources/text_det.jpg')
# create text detector
detector = TextDetector(
    model_path='mmdeploy_models/mmocr/dbnet/ort',
    device_name='cpu',
    device_id=0)
# do model inference
bboxes = detector(img)
# draw detected bbox into the input image
if len(bboxes) > 0:
    pts = ((bboxes[:, 0:8] + 0.5).reshape(len(bboxes), -1,
                                          2).astype(int))
    cv2.polylines(img, pts, True, (0, 255, 0), 2)
    cv2.imwrite('output_ocr.png', img)
```

#### Text Recognition SDK model inference

```python
import cv2
from mmdeploy_runtime import TextRecognizer

img = cv2.imread('demo/resources/text_recog.jpg')
# create text recognizer
recognizer = TextRecognizer(
  model_path='mmdeploy_models/mmocr/crnn/ort',
  device_name='cpu',
  device_id=0
)
# do model inference
texts = recognizer(img)
# print the result
print(texts)
```

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/main/demo).

## Supported models

| Model                                                                                | Task             | TorchScript | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVINO |
| :----------------------------------------------------------------------------------- | :--------------- | :---------: | :---------: | :------: | :--: | :---: | :------: |
| [DBNet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet)         | text-detection   |      Y      |      Y      |    Y     |  Y   |   Y   |    Y     |
| [DBNetpp](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnetpp)     | text-detection   |      N      |      Y      |    Y     |  ?   |   ?   |    Y     |
| [PSENet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet)       | text-detection   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [PANet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/panet)         | text-detection   |      Y      |      Y      |    Y     |  Y   |   N   |    Y     |
| [TextSnake](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/textsnake) | text-detection   |      Y      |      Y      |    Y     |  ?   |   ?   |    ?     |
| [MaskRCNN](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/maskrcnn)   | text-detection   |      Y      |      Y      |    Y     |  ?   |   ?   |    ?     |
| [CRNN](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn)         | text-recognition |      Y      |      Y      |    Y     |  Y   |   Y   |    N     |
| [SAR](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/sar)           | text-recognition |      N      |      Y      |    Y     |  N   |   N   |    N     |
| [SATRN](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/satrn)       | text-recognition |      Y      |      Y      |    Y     |  N   |   N   |    N     |
| [ABINet](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/abinet)     | text-recognition |      Y      |      Y      |    Y     |  ?   |   ?   |    ?     |

## Reminder

- ABINet for TensorRT require pytorch1.10+ and TensorRT 8.4+.

- SAR uses `valid_ratio` inside network inference, which causes performance drops. When the `valid_ratio`s between
  testing image and the image for conversion are quite different, the gap would be enlarged.

- For TensorRT backend, users have to choose the right config. For example, CRNN only accepts 1 channel input. Here is a recommendation table:

  | Model    | Config                                                     |
  | :------- | :--------------------------------------------------------- |
  | MaskRCNN | text-detection_mrcnn_tensorrt_dynamic-320x320-2240x2240.py |
  | CRNN     | text-recognition_tensorrt_dynamic-1x32x32-1x32x640.py      |
  | SATRN    | text-recognition_tensorrt_dynamic-32x32-32x640.py          |
  | SAR      | text-recognition_tensorrt_dynamic-48x64-48x640.py          |
  | ABINet   | text-recognition_tensorrt_static-32x128.py                 |
