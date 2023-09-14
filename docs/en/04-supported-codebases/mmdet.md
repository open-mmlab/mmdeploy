# MMDetection Deployment

- [MMDetection Deployment](#mmdetection-deployment)
  - [Installation](#installation)
    - [Install mmdet](#install-mmdet)
    - [Install mmdeploy](#install-mmdeploy)
  - [Convert model](#convert-model)
  - [Model specification](#model-specification)
  - [Model inference](#model-inference)
    - [Backend model inference](#backend-model-inference)
    - [SDK model inference](#sdk-model-inference)
  - [Supported models](#supported-models)
  - [Reminder](#reminder)

______________________________________________________________________

[MMDetection](https://github.com/open-mmlab/mmdetection) aka `mmdet` is an open source object detection toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

## Installation

### Install mmdet

Please follow the [installation guide](https://mmdetection.readthedocs.io/en/3.x/get_started.html) to install mmdet.

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

You can use [tools/deploy.py](https://github.com/open-mmlab/mmdeploy/tree/main/tools/deploy.py) to convert mmdet models to the specified backend models. Its detailed usage can be learned from [here](../02-how-to-run/convert_model.md).

The command below shows an example about converting `Faster R-CNN` model to onnx model that can be inferred by ONNX Runtime.

```shell
cd mmdeploy
# download faster r-cnn model from mmdet model zoo
mim download mmdet --config faster-rcnn_r50_fpn_1x_coco --dest .
# convert mmdet model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmdet/detection/detection_onnxruntime_dynamic.py \
    faster-rcnn_r50_fpn_1x_coco.py \
    faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    demo/resources/det.jpg \
    --work-dir mmdeploy_models/mmdet/ort \
    --device cpu \
    --show \
    --dump-info
```

It is crucial to specify the correct deployment config during model conversion. We've already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmdet) of all supported backends for mmdetection, under which the config file path follows the pattern:

```
{task}/{task}_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{task}:** task in mmdetection.

  There are two of them. One is `detection` and the other is `instance-seg`, indicating instance segmentation.

  mmdet models like `RetinaNet`, `Faster R-CNN` and `DETR` and so on belongs to `detection` task. While `Mask R-CNN` is one of `instance-seg` models. You can find more of them in chapter [Supported models](#supported-models).

  **DO REMEMBER TO USE** `detection/detection_*.py` deployment config file when trying to convert detection models and use `instance-seg/instance-seg_*.py` to deploy instance segmentation models.

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml etc.

- **{precision}:** fp16, int8. When it's empty, it means fp32

- **{static | dynamic}:** static shape or dynamic shape

- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `faster r-cnn` to other backend models by changing the deployment config file `detection_onnxruntime_dynamic.py` to [others](https://github.com/open-mmlab/mmdeploy/tree/main/configs/mmdet/detection), e.g., converting to tensorrt-fp16 model by `detection_tensorrt-fp16_dynamic-320x320-1344x1344.py`.

```{tip}
When converting mmdet models to tensorrt models, --device should be set to "cuda"
```

## Model specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmdet/ort` in the previous example. It includes:

```
mmdeploy_models/mmdet/ort
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- \***.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmdet/ort** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model and visualize the results.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmdet/detection/detection_onnxruntime_dynamic.py'
model_cfg = './faster-rcnn_r50_fpn_1x_coco.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmdet/ort/end2end.onnx']
image = './demo/resources/det.jpg'

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
    output_file='output_detection.png')
```

### SDK model inference

You can also perform SDK model inference like following,

```python
from mmdeploy_runtime import Detector
import cv2

img = cv2.imread('./demo/resources/det.jpg')

# create a detector
detector = Detector(model_path='./mmdeploy_models/mmdet/ort', device_name='cpu', device_id=0)
# perform inference
bboxes, labels, masks = detector(img)

# visualize inference result
indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
  [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
  if score < 0.3:
    continue

  cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imwrite('output_detection.png', img)
```

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/main/demo).

## Supported models

|                                                        Model                                                        |         Task          | OnnxRuntime | TensorRT | ncnn | PPLNN | OpenVINO |
| :-----------------------------------------------------------------------------------------------------------------: | :-------------------: | :---------: | :------: | :--: | :---: | :------: |
|                      [ATSS](https://github.com/open-mmlab/mmdetection/tree/main/configs/atss)                       |   Object Detection    |      Y      |    Y     |  N   |   N   |    Y     |
|                      [FCOS](https://github.com/open-mmlab/mmdetection/tree/main/configs/fcos)                       |   Object Detection    |      Y      |    Y     |  Y   |   N   |    Y     |
|                  [FoveaBox](https://github.com/open-mmlab/mmdetection/tree/main/configs/foveabox)                   |   Object Detection    |      Y      |    N     |  N   |   N   |    Y     |
|                      [FSAF](https://github.com/open-mmlab/mmdetection/tree/main/configs/fsaf)                       |   Object Detection    |      Y      |    Y     |  Y   |   Y   |    Y     |
|                 [RetinaNet](https://github.com/open-mmlab/mmdetection/tree/main/configs/retinanet)                  |   Object Detection    |      Y      |    Y     |  Y   |   Y   |    Y     |
|                       [SSD](https://github.com/open-mmlab/mmdetection/tree/main/configs/ssd)                        |   Object Detection    |      Y      |    Y     |  Y   |   N   |    Y     |
|                     [VFNet](https://github.com/open-mmlab/mmdetection/tree/main/configs/vfnet)                      |   Object Detection    |      N      |    N     |  N   |   N   |    Y     |
|                     [YOLOv3](https://github.com/open-mmlab/mmdetection/tree/main/configs/yolo)                      |   Object Detection    |      Y      |    Y     |  Y   |   N   |    Y     |
|                     [YOLOX](https://github.com/open-mmlab/mmdetection/tree/main/configs/yolox)                      |   Object Detection    |      Y      |    Y     |  Y   |   N   |    Y     |
|              [Cascade R-CNN](https://github.com/open-mmlab/mmdetection/tree/main/configs/cascade_rcnn)              |   Object Detection    |      Y      |    Y     |  N   |   Y   |    Y     |
|               [Faster R-CNN](https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn)               |   Object Detection    |      Y      |    Y     |  Y   |   Y   |    Y     |
|            [Faster R-CNN + DCN](https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn)            |   Object Detection    |      Y      |    Y     |  Y   |   Y   |    Y     |
|                       [GFL](https://github.com/open-mmlab/mmdetection/tree/main/configs/gfl)                        |   Object Detection    |      Y      |    Y     |  N   |   ?   |    Y     |
|                 [RepPoints](https://github.com/open-mmlab/mmdetection/tree/main/configs/reppoints)                  |   Object Detection    |      N      |    Y     |  N   |   ?   |    Y     |
|             [DETR](https://github.com/open-mmlab/mmdetection/tree/main/configs/detr)[\*](#nobatchinfer)             |   Object Detection    |      Y      |    Y     |  N   |   ?   |    Y     |
|  [Deformable DETR](https://github.com/open-mmlab/mmdetection/tree/main/configs/deformable_detr)[\*](#nobatchinfer)  |   Object Detection    |      Y      |    Y     |  N   |   ?   |    Y     |
| [Conditional DETR](https://github.com/open-mmlab/mmdetection/tree/main/configs/conditional_detr)[\*](#nobatchinfer) |   Object Detection    |      Y      |    Y     |  N   |   ?   |    Y     |
|         [DAB-DETR](https://github.com/open-mmlab/mmdetection/tree/main/configs/dab_detr)[\*](#nobatchinfer)         |   Object Detection    |      Y      |    Y     |  N   |   ?   |    Y     |
|             [DINO](https://github.com/open-mmlab/mmdetection/tree/main/configs/dino)[\*](#nobatchinfer)             |   Object Detection    |      Y      |    Y     |  N   |   ?   |    Y     |
|                 [CenterNet](https://github.com/open-mmlab/mmdetection/tree/main/configs/centernet)                  |   Object Detection    |      Y      |    Y     |  N   |   ?   |    Y     |
|                    [RTMDet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet)                     |   Object Detection    |      Y      |    Y     |  N   |   ?   |    Y     |
|           [Cascade Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/main/configs/cascade_rcnn)            | Instance Segmentation |      Y      |    Y     |  N   |   N   |    Y     |
|                       [HTC](https://github.com/open-mmlab/mmdetection/tree/main/configs/htc)                        | Instance Segmentation |      Y      |    Y     |  N   |   ?   |    Y     |
|                 [Mask R-CNN](https://github.com/open-mmlab/mmdetection/tree/main/configs/mask_rcnn)                 | Instance Segmentation |      Y      |    Y     |  N   |   N   |    Y     |
|                [Swin Transformer](https://github.com/open-mmlab/mmdetection/tree/main/configs/swin)                 | Instance Segmentation |      Y      |    Y     |  N   |   N   |    Y     |
|                      [SOLO](https://github.com/open-mmlab/mmdetection/tree/main/configs/solo)                       | Instance Segmentation |      Y      |    N     |  N   |   N   |    Y     |
|                    [SOLOv2](https://github.com/open-mmlab/mmdetection/tree/main/configs/solov2)                     | Instance Segmentation |      Y      |    N     |  N   |   N   |    Y     |
|                  [CondInst](https://github.com/open-mmlab/mmdetection/tree/main/configs/condinst)                   | Instance Segmentation |      Y      |    Y     |  N   |   N   |    N     |
|              [Panoptic FPN](https://github.com/open-mmlab/mmdetection/tree/main/configs/panoptic_fpn)               | Panoptic Segmentation |      Y      |    Y     |  N   |   N   |    N     |
|                [MaskFormer](https://github.com/open-mmlab/mmdetection/tree/main/configs/maskformer)                 | Panoptic Segmentation |      Y      |    Y     |  N   |   N   |    N     |
|      [Mask2Former](https://github.com/open-mmlab/mmdetection/tree/main/configs/mask2former)[\*](#mask2former)       | Panoptic Segmentation |      Y      |    Y     |  N   |   N   |    N     |

## Reminder

- For transformer based models, strongly suggest use `TensorRT>=8.4`.
- <i id="mask2former">Mask2Former</i> should use `TensorRT>=8.6.1` for dynamic shape inference.
- <i id="nobatchinfer">DETR-like models</i> do not support multi-batch inference.
