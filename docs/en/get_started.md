## Get Started

MMDeploy provides some useful tools. It is easy to deploy models in OpenMMLab to various platforms. You can convert models in our pre-defined pipeline or build a custom conversion pipeline by yourself. This guide will show you how to convert a model with MMDeploy and integrate MMDeploy's SDK to your application!

### Prerequisites

First we should install MMDeploy following [build.md](./build.md). Note that the build steps are slightly different among the supported backends. Here are some brief introductions to these backends:

- [ONNXRuntime](./backends/onnxruntime.md): ONNX Runtime is a cross-platform inference and training machine-learning accelerator. It has best support for <span style="color:red">ONNX IR</span>.
- [TensorRT](./backends/tensorrt.md): NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning inference applications. It is a good choice if you want to deploy your model on <span style="color:red">NVIDIA devices</span>.
- [ncnn](./backends/ncnn.md): ncnn is a high-performance neural network inference computing framework optimized for <span style="color:red">mobile platforms</span>. ncnn is deeply considerate about deployment and uses on <span style="color:red">mobile phones</span> from the beginning of design.
- [PPLNN](./backends/pplnn.md): PPLNN, which is short for "PPLNN is a Primitive Library for Neural Network", is a high-performance deep-learning inference engine for efficient AI inferencing. It can run various ONNX models and has <span style="color:red">better support for OpenMMLab</span>.
- [OpenVINO](./backends/openvino.md): OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference. The open-source toolkit allows to seamlessly integrate with <span style="color:red">Intel AI hardware</span>, the latest neural network accelerator chips, the Intel AI stick, and embedded computers or edge devices.

Choose the backend which can meet your demand and install it following the link provided above.

### Convert Model

Once you have installed MMDeploy, you can convert the PyTorch model in the OpenMMLab model zoo to the backend model with one magic spell! For example, if you want to convert the Faster-RCNN in [MMDetection](https://github.com/open-mmlab/mmdetection) to TensorRT:

```bash
# Assume you have installed MMDeploy in ${MMDEPLOY_DIR} and MMDetection in ${MMDET_DIR}
# If you do not know where to find the path. Just type `pip show mmdeploy` and `pip show mmdet` in your console.

python ${MMDEPLOY_DIR}/tools/deploy.py \
    ${MMDEPLOY_DIR}/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    ${CHECKPOINT_DIR}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ${INPUT_IMG} \
    --work-dir ${WORK_DIR} \
    --device cuda:0 \
    --dump-info
```

`${MMDEPLOY_DIR}/tools/deploy.py` is a tool that does everything you need to convert a model. Read [how_to_convert_model](./tutorials/how_to_convert_model.md) for more details. The converted model and other meta-info will be found in `${WORK_DIR}`. And they make up of MMDeploy SDK Model that can be fed to MMDeploy SDK to do model inference.

`detection_tensorrt_dynamic-320x320-1344x1344.py` is a config file that contains all arguments you need to customize the conversion pipeline. The name is formed as

```bash
<task name>_<backend>-[backend options]_<dynamic support>.py
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
    ${MMDEPLOY_DIR}/configs/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --model ${BACKEND_MODEL_FILES} \
    --metrics ${METRICS} \
    --device cuda:0
```

Read [how to evaluate a model](./tutorials/how_to_evaluate_a_model.md) for more details about how to use `tools/test.py`

### Integrate MMDeploy SDK

Make sure to turn on `MMDEPLOY_BUILD_SDK` to build and install SDK by following [build.md](./build.md).
After that, the structure in the installation folder will show as follows,

```
install
├── example
├── include
│   ├── c
│   └── cpp
└── lib
```
where `include/c` and `include/cpp` correspond to C and C++ API respectively.

**Caution: The C++ API is highly volatile and not recommended at the moment.**

In the example directory, there are several examples involving classification, object detection, image segmentation and so on.
You can refer to these examples to learn how to use MMDeploy SDK's C API and how to link ${MMDeploy_LIBS} to your application.

### A From-scratch Example

Here is an example of how to deploy and inference Faster R-CNN model of MMDetection from scratch.

#### Create Virtual Environment and Install MMDetection.

Please run the following command in Anaconda environment to [install MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#a-from-scratch-setup-script).

```bash
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch -y

# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html

# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

#### Download the Checkpoint of Faster R-CNN

Download the checkpoint from this [link](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) and put it in the `{MMDET_ROOT}/checkpoints` where `{MMDET_ROOT}` is the root directory of your MMDetection codebase.

#### Install MMDeploy and ONNX Runtime

Please run the following command in Anaconda environment to [install MMDeploy](./build.md).
```bash
conda activate openmmlab

git clone https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
git submodule update --init --recursive
pip install -e .
```

Once we have installed the MMDeploy, we should select an inference engine for model inference. Here we take ONNX Runtime as an example. Run the following command to [install ONNX Runtime](./backends/onnxruntime.md):
```bash
pip install onnxruntime==1.8.1
```

Then download the ONNX Runtime library to build the mmdeploy plugin for ONNX Runtime:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz

tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
cd onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=$(pwd)
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

cd ${MMDEPLOY_DIR} # To MMDeploy root directory
mkdir -p build && cd build

# build ONNXRuntime custom ops
cmake -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc)

# build MMDeploy SDK
cmake -DMMDEPLOY_BUILD_SDK=ON \
      -DCMAKE_CXX_COMPILER=g++-7 \
      -DOpenCV_DIR=/path/to/OpenCV/lib/cmake/OpenCV \
      -Dspdlog_DIR=/path/to/spdlog/lib/cmake/spdlog \
      -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
      -DMMDEPLOY_TARGET_BACKENDS=ort \
      -DMMDEPLOY_CODEBASES=mmdet ..
make -j$(nproc) && make install
```

#### Model Conversion

Once we have installed MMDetection, MMDeploy, ONNX Runtime and built plugin for ONNX Runtime, we can convert the Faster R-CNN to a `.onnx` model file which can be received by ONNX Runtime. Run following commands to use our deploy tools:

```bash
# Assume you have installed MMDeploy in ${MMDEPLOY_DIR} and MMDetection in ${MMDET_DIR}
# If you do not know where to find the path. Just type `pip show mmdeploy` and `pip show mmdet` in your console.

python ${MMDEPLOY_DIR}/tools/deploy.py \
    ${MMDEPLOY_DIR}/configs/mmdet/detection/detection_onnxruntime_dynamic.py \
    ${MMDET_DIR}/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    ${MMDET_DIR}/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ${MMDET_DIR}/demo/demo.jpg \
    --work-dir work_dirs \
    --device cpu \
    --show \
    --dump-info
```

If the script runs successfully, two images will display on the screen one by one. The first image is the infernce result of ONNX Runtime and the second image is the result of PyTorch. At the same time, an onnx model file `end2end.onnx` and three json files (SDK config files) will generate on the work directory `work_dirs`.

#### Run MMDeploy SDK demo

After model conversion, SDK Model is saved in directory ${work_dir}.
Here is a recipe for building & running object detection demo.
```Bash
cd build/install/example

# path to onnxruntime ** libraries **
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib

mkdir -p build && cd build
cmake -DOpenCV_DIR=path/to/OpenCV/lib/cmake/OpenCV \
      -DMMDeploy_DIR=${MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy ..
make object_detection

# suppress verbose logs
export SPDLOG_LEVEL=warn

# running the object detection example
./object_detection cpu ${work_dirs} ${path/to/an/image}

```
If the demo runs successfully, an image named "output_detection.png" is supposed to be found showing detection objects.

### Add New Model Support?

If the models you want to deploy have not been supported yet in MMDeploy, you can try to support them by yourself. Here are some documents that may help you:
- Read [how_to_support_new_models](./tutorials/how_to_support_new_models.md) to learn more about the rewriter.




Finally, we welcome your PR!
