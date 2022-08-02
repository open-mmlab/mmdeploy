# Get Started

MMDeploy provides useful tools for deploying OpenMMLab models to various platforms and devices.

With the help of them, you can not only do model deployment using our pre-defined pipelines but also customize your own deployment pipeline.

## Introduction

In MMDeploy, the deployment pipeline can be illustrated by a sequential modules, i.e., Model Converter, MMDeploy Model and Inference SDK.

![deploy-pipeline](https://user-images.githubusercontent.com/4560679/172306700-31b4c922-2f04-42ed-a1d6-c360f2f3048c.png)

### Model Converter

Model Converter aims at converting training models from OpenMMLab into backend models that can be run on target devices.
It is able to transform PyTorch model into IR model, i.e., ONNX, TorchScript, as well as convert IR model to backend model. By combining them together, we can achieve one-click **end-to-end** model deployment.

### MMDeploy Model

MMDeploy Model is the result package exported by Model Converter.
Beside the backend models, it also includes the model meta info, which will be used by Inference SDK.

### Inference SDK

Inference SDK is developed by C/C++, wrapping the preprocessing, model forward and postprocessing modules in model inference.
It supports FFI such as C, C++, Python, C#, Java and so on.

## Prerequisites

In order to do an end-to-end model deployment, MMDeploy requires Python 3.6+ and PyTorch 1.5+.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name mmdeploy python=3.8 -y
conda activate mmdeploy
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch=={pytorch_version} torchvision=={torchvision_version} cudatoolkit={cudatoolkit_version} -c pytorch -c conda-forge
```

On CPU platforms:

```shell
conda install pytorch=={pytorch_version} torchvision=={torchvision_version} cpuonly -c pytorch
```

```{note}
On GPU platform, please ensure that {cudatoolkit_version} matches your host CUDA toolkit version. Otherwise, it probably brings in conflicts when deploying model with TensorRT.
```

## Installation

We recommend that users follow our best practices installing MMDeploy.

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMDeploy and inference engine

We recommend using MMDeploy precompiled package as our best practice.
You can download them from [here](https://github.com/open-mmlab/mmdeploy/releases) according to your target platform and device.

The supported platform and device matrix is presented as following:

<table>
<thead>
  <tr>
    <th>OS-Arch</th>
    <th>Device</th>
    <th>ONNX Runtime</th>
    <th>TensorRT</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Linux-x86_64</td>
    <td>CPU</td>
    <td>Y</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>N</td>
    <td>Y</td>
  </tr>
  <tr>
    <td rowspan="2">Windows-x86_64</td>
    <td>CPU</td>
    <td>Y</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>N</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**Note: if MMDeploy prebuilt package doesn't meet your target platforms or devices, please [build MMDeploy from source](./01-how-to-build/build_from_source.md)**

Take the latest precompiled package as example, you can install it as follows:

<details open>
<summary><b>Linux-x86_64, CPU, ONNX Runtime 1.8.1</b></summary>

```shell
  # install MMDeploy
wget https://github.com/open-mmlab/mmdeploy/releases/download/v0.7.0/mmdeploy-0.7.0-linux-x86_64-onnxruntime1.8.1.tar.gz
tar -zxvf mmdeploy-0.7.0-linux-x86_64-onnxruntime1.8.1.tar.gz
cd mmdeploy-0.7.0-linux-x86_64-onnxruntime1.8.1
pip install dist/mmdeploy-0.7.0-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-0.7.0-cp38-none-linux_x86_64.whl
cd ..
# install inference engine: ONNX Runtime
pip install onnxruntime==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

</details>

<details open>
<summary><b>Linux-x86_64, CUDA 11.x, TensorRT 8.2.3.0</b></summary>

```shell
# install MMDeploy
wget https://github.com/open-mmlab/mmdeploy/releases/download/v0.7.0/mmdeploy-0.7.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0.tar.gz
tar -zxvf mmdeploy-v0.7.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0.tar.gz
cd mmdeploy-0.7.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0
pip install dist/mmdeploy-0.7.0-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-0.7.0-cp38-none-linux_x86_64.whl
cd ..
# install inference engine: TensorRT
# !!! Download TensorRT-8.2.3.0 CUDA 11.x tar package from NVIDIA, and extract it to the current directory
pip install TensorRT-8.2.3.0/python/tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl
pip install pycuda
export TENSORRT_DIR=$(pwd)/TensorRT-8.2.3.0
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
# !!! Download cuDNN 8.2.1 CUDA 11.x tar package from NVIDIA, and extract it to the current directory
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
```

</details>

<details>
<summary><b>Windows, CPU, ONNX Runtime 1.8.1</b></summary>

Please open `Anaconda Powershell Prompt` and run the following commands:

```shell
# install MMDeploy
Invoke-WebRequest -Uri https://github.com/open-mmlab/mmdeploy/releases/download/v0.7.0/mmdeploy-0.7.0-windows-amd64-onnxruntime1.8.1.zip -OutFile mmdeploy-0.7.0-windows-amd64-onnxruntime1.8.1.zip
Expand-Archive mmdeploy-0.7.0-windows-amd64-onnxruntime1.8.1.zip .
cd  mmdeploy-0.7.0-windows-amd64-onnxruntime1.8.1
pip install dist/mmdeploy-0.7.0-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-0.7.0-cp38-none-linux_x86_64.whl
cd ..

# install inference engine: ONNX Runtime
pip install onnxruntime==1.8.1
Invoke-WebRequest -Uri https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-win-x64-1.8.1.zip -OutFile onnxruntime-win-x64-1.8.1.zip
Expand-Archive onnxruntime-win-x64-1.8.1.zip .
$env:ONNXRUNTIME_DIR=$(pwd)/onnxruntime-win-x64-1.8.1
$env:path="$env:ONNXRUNTIME_DIR"/lib:$env:path
```

</details>

<details>
<summary><b>Windows, CUDA 11.x, TensorRT 8.2.3.0</b></summary>

请在 Conda Prompt Shell 中执行以下命令

```shell
# install MMDeploy
Invoke-WebRequest -Uri https://github.com/open-mmlab/mmdeploy/releases/download/v0.7.0/mmdeploy-0.7.0-windows-amd64-cuda11.1-tensorrt8.2.3.0.zip -OutFile mmdeploy-0.7.0-windows-amd64-cuda11.1-tensorrt8.2.3.0.zip
Expand-Archive mmdeploy-0.7.0-windows-amd64-cuda11.1-tensorrt8.2.3.0.zip .
cd mmdeploy-0.7.0-windows-amd64-cuda11.1-tensorrt8.2.3.0
pip install dist/mmdeploy-0.7.0-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-0.7.0-cp38-none-linux_x86_64.whl

# install inference engine: TensorRT
# !!! Download TensorRT-8.2.3.0 CUDA 11.x tar package from NVIDIA, and extract it to the current directory
pip install TensorRT-8.2.3.0/python/tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl
pip install pycuda
$env:TENSORRT_DIR=$(pwd)/TensorRT-8.2.3.0
$env:path="$env:TENSORRT_DIR"/lib:$env:path
# !!! Download cuDNN 8.2.1 CUDA 11.x tar package from NVIDIA, and extract it to the current directory
$env:CUDNN_DIR=$(pwd)/cuda
$env:path="$env:CUDNN_DIR"/lib64:$env:path
```

</details>

## Convert Model

After the installation, you can enjoy the model deployment journey starting from converting PyTorch model to backend model by running `tools/deploy.py`.

Based on the above settings, we provide an example to convert the Faster R-CNN in [MMDetection](https://github.com/open-mmlab/mmdetection) to TensorRT as below:

```shell
# clone mmdeploy to get the deployment config. `--recursive` is not necessary
git clone https://github.com/open-mmlab/mmdeploy.git

# clone mmdetection repo. We have to use the config file to build PyTorch nn module
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ..

# download Faster R-CNN checkpoint
wget -P checkpoints https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# run the command to start model conversion
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    mmdetection/demo/demo.jpg \
    --work-dir mmdeploy_model/faster-rcnn \
    --device cuda \
    --dump-info
```

The converted model and its meta info will be found in the path specified by `--work-dir`.
And they make up of MMDeploy Model that can be fed to MMDeploy SDK to do model inference.

For more details about model conversion, you can read [how_to_convert_model](./02-how-to-run/convert_model.md). If you want to customize the conversion pipeline, you can edit the config file by following [this](./02-how-to-run/write_config.md) tutorial.

```{tip}
If MMDeploy-ONNXRuntime prebuild package is installed, you can convert the above model to onnx model and perform ONNX Runtime inference
just by 'changing detection_tensorrt_dynamic-320x320-1344x1344.py' to 'detection_onnxruntime_dynamic.py' and making '--device' as 'cpu'.
```

## Inference Model

After model conversion, we can perform inference not only by Model Converter but also by Inference SDK.

### Inference by Model Converter

Model Converter provides a unified API named as `inference_model` to do the job, making all inference backends API transparent to users.
Take the previous converted Faster R-CNN tensorrt model for example,

```python
from mmdeploy.apis import inference_model
result = inference_model(
  model_cfg='mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
  deploy_cfg='mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py',
  backend_files=['mmdeploy_model/faster-rcnn/end2end.engine'],
  img='mmdetection/demo/demo.jpg',
  device='cuda:0')
```

```{note}
'backend_files' in this API refers to backend engine file path, which MUST be put in a list, since some inference engines like OpenVINO and ncnn separate the network structure and its weights into two files.
```

### Inference by SDK

You can directly run MMDeploy demo programs in the precompiled package to get inference results.

```shell
cd mmdeploy-0.7.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0
# run python demo
python sdk/example/python/object_detection.py cuda ../mmdeploy_model/faster-rcnn ../mmdetection/demo/demo.jpg
# run C/C++ demo
export LD_LIBRARY_PATH=$(pwd)/sdk/lib:$LD_LIBRARY_PATH
./sdk/bin/object_detection cuda ../mmdeploy_model/faster-rcnn ../mmdetection/demo/demo.jpg
```

```{note}
In the above command, the input model is SDK Model path. It is NOT engine file path but actually the path passed to --work-dir. It not only includes engine files but also meta information like 'deploy.json' and 'pipeline.json'.
```

In the next section, we will provide examples of deploying the converted Faster R-CNN model talked above with SDK different FFI (Foreign Function Interface).

#### Python API

```python
from mmdeploy_python import Detector
import cv2

img = cv2.imread('mmdetection/demo/demo.jpg')

# create a detector
detector = Detector(model_path='mmdeploy_models/faster-rcnn', device_name='cuda', device_id=0)
# run the inference
bboxes, labels, _ = detector(img)
# Filter the result according to threshold
indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
  [left, top, right, bottom], score = bbox[0:4].astype(int),  bbox[4]
  if score < 0.3:
      continue
  cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imwrite('output_detection.png', img)
```

You can find more examples from [here](https://github.com/open-mmlab/mmdeploy/tree/master/demo/python).

#### C API

Using SDK C API should follow next pattern,

```{mermaid}
graph LR
  A[create inference handle] --> B(read image)
  B --> C(apply handle)
  C --> D[deal with inference result]
  D -->E[destroy result buffer]
  E -->F[destroy handle]
```

Now let's apply this procedure on the above Faster R-CNN model.

```C++
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "detector.h"

int main() {
  const char* device_name = "cuda";
  int device_id = 0;

  std::string model_path = "mmdeploy_model/faster-rcnn";
  std::string image_path = "mmdetection/demo/demo.jpg";

  // create inference handle
  mmdeploy_detector_t detector{};
  int status{};
  status = mmdeploy_detector_create_by_path(model_path, device_name, device_id, &detector);
  assert(status == MMDEPLOY_SUCCESS);

  // read image
  cv::Mat img = cv::imread(image_path);
  assert(img.data);

  // apply handle and get the inference result
  mmdeploy_mat_t mat{img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};
  mmdeploy_detection_t *bboxes{};
  int *res_count{};
  status = mmdeploy_detector_apply(detector, &mat, 1, &bboxes, &res_count);
  assert (status == MMDEPLOY_SUCCESS);

  // deal with the result. Here we choose to visualize it
  for (int i = 0; i < *res_count; ++i) {
    const auto &box = bboxes[i].bbox;
    if (bboxes[i].score < 0.3) {
      continue;
    }
    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }

  cv::imwrite("output_detection.png", img);

  // destroy result buffer
  mmdeploy_detector_release_result(bboxes, res_count, 1);
  // destroy inference handle
  mmdeploy_detector_destroy(detector);
  return 0;
}
```

When you build this example, try to add MMDeploy package in your CMake project as following. Then pass `-DMMDeploy_DIR` to cmake, which indicates the path where `MMDeployConfig.cmake` locates. You can find it in the prebuilt package.

```Makefile
find_package(MMDeploy REQUIRED)
target_link_libraries(${name} PRIVATE mmdeploy ${OpenCV_LIBS})
```

For more SDK C API usages, please read these [samples](https://github.com/open-mmlab/mmdeploy/tree/master/demo/csrc).

#### C++ API

You can learn the usage of MMDeploy C++ API from [demo codes](https://github.com/open-mmlab/mmdeploy/tree/master/demo/csrc). We'll talk about them more in our next release.

#### C# API

You can learn the usage of MMDeploy C+ API from [demo codes](https://github.com/open-mmlab/mmdeploy/tree/master/demo/csharp). We'll talk about them more in our next release.

#### JAVA API

You can learn the usage of MMDeploy C+ API from [demo codes](https://github.com/open-mmlab/mmdeploy/tree/master/demo/java). We'll talk about them more in our next release.

## Evaluate Model

You can test the performance of deployed model using `tool/test.py`. For example,

```shell
python ${MMDEPLOY_DIR}/tools/test.py \
    ${MMDEPLOY_DIR}/configs/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --model ${BACKEND_MODEL_FILES} \
    --metrics ${METRICS} \
    --device cuda:0
```

```{note}
Regarding the --model option, it represents the converted engine files path when using Model Converter to do performance test. But when you try to test the metrics by Inference SDK, this option refers to the directory path of MMDeploy Model.
```

You can read [how to evaluate a model](02-how-to-run/how_to_evaluate_a_model.md) for more details.
