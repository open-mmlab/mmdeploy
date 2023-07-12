# OpenVINO Support

This tutorial is based on Linux systems like Ubuntu-18.04.

## Installation

It is recommended to create a virtual environment for the project.

### Install python package

Install [OpenVINO](https://docs.openvino.ai/2022.3/get_started.html). It is recommended to use the installer or install using pip.
Installation example using [pip](https://pypi.org/project/openvino-dev/):

```bash
pip install openvino-dev[onnx]==2022.3.0
```

### Download OpenVINO runtime for SDK (Optional)

If you want to use OpenVINO in SDK, you need install OpenVINO with [install_guides](https://docs.openvino.ai/2022.3/openvino_docs_install_guides_installing_openvino_from_archive_linux.html#installing-openvino-runtime).
Take `openvino==2022.3.0` as example:

```bash
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3/linux/l_openvino_toolkit_ubuntu20_2022.3.0.9052.9752fafe8eb_x86_64.tgz
tar xzf ./l_openvino_toolkit*.tgz
cd l_openvino*
export InferenceEngine_DIR=$pwd/runtime/cmake
bash ./install_dependencies/install_openvino_dependencies.sh
```

### Build mmdeploy SDK with OpenVINO (Optional)

Install MMDeploy following the [instructions](../01-how-to-build/build_from_source.md).

```bash
cd ${MMDEPLOY_DIR} # To MMDeploy root directory
mkdir -p build && cd build
cmake -DMMDEPLOY_TARGET_DEVICES='cpu' -DMMDEPLOY_TARGET_BACKENDS=openvino -DInferenceEngine_DIR=${InferenceEngine_DIR} ..
make -j$(nproc) && make install
```

To work with models from [MMDetection](https://mmdetection.readthedocs.io/en/3.x/get_started.html), you may need to install it additionally.

## Usage

You could follow the instructions of tutorial [How to convert model](../02-how-to-run/convert_model.md)

Example:

```bash
python tools/deploy.py \
    configs/mmdet/detection/detection_openvino_static-300x300.py \
    /mmdetection_dir/mmdetection/configs/ssd/ssd300_coco.py \
    /tmp/snapshots/ssd300_coco_20210803_015428-d231a06e.pth \
    tests/data/tiger.jpeg \
    --work-dir ../deploy_result \
    --device cpu \
    --log-level INFO
```

## List of supported models exportable to OpenVINO from MMDetection

The table below lists the models that are guaranteed to be exportable to OpenVINO from MMDetection.

|     Model name     |                                  Config                                   | Dynamic Shape |
| :----------------: | :-----------------------------------------------------------------------: | :-----------: |
|        ATSS        |                  `configs/atss/atss_r50_fpn_1x_coco.py`                   |       Y       |
| Cascade Mask R-CNN |        `configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py`        |       Y       |
|   Cascade R-CNN    |          `configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py`           |       Y       |
|    Faster R-CNN    |           `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`            |       Y       |
|        FCOS        | `configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco.py` |       Y       |
|      FoveaBox      |             `configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py `              |       Y       |
|        FSAF        |                  `configs/fsaf/fsaf_r50_fpn_1x_coco.py`                   |       Y       |
|     Mask R-CNN     |             `configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py`              |       Y       |
|     RetinaNet      |             `configs/retinanet/retinanet_r50_fpn_1x_coco.py`              |       Y       |
|        SSD         |                       `configs/ssd/ssd300_coco.py`                        |       Y       |
|       YOLOv3       |            `configs/yolo/yolov3_d53_mstrain-608_273e_coco.py`             |       Y       |
|       YOLOX        |                `configs/yolox/yolox_tiny_8x8_300e_coco.py`                |       Y       |
| Faster R-CNN + DCN |         `configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py`          |       Y       |
|       VFNet        |                 `configs/vfnet/vfnet_r50_fpn_1x_coco.py`                  |       Y       |

Notes:

- Custom operations from OpenVINO use the domain `org.openvinotoolkit`.
- For faster work in OpenVINO in the Faster-RCNN, Mask-RCNN, Cascade-RCNN, Cascade-Mask-RCNN models
  the RoiAlign operation is replaced with the [ExperimentalDetectronROIFeatureExtractor](https://docs.openvino.ai/2022.3/openvino_docs_ops_detection_ExperimentalDetectronROIFeatureExtractor_6.html) operation in the ONNX graph.
- Models "VFNet" and "Faster R-CNN + DCN" use the custom "DeformableConv2D" operation.

## Deployment config

With the deployment config, you can specify additional options for the Model Optimizer.
To do this, add the necessary parameters to the `backend_config.mo_options` in the fields `args` (for parameters with values) and `flags` (for flags).

Example:

```python
backend_config = dict(
    mo_options=dict(
        args=dict({
            '--mean_values': [0, 0, 0],
            '--scale_values': [255, 255, 255],
            '--data_type': 'FP32',
        }),
        flags=['--disable_fusing'],
    )
)
```

Information about the possible parameters for the Model Optimizer can be found in the [documentation](https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html).

## Troubleshooting

- ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory

  To resolve missing external dependency on Ubuntu\*, execute the following command:

  ```bash
  sudo apt-get install libpython3.7
  ```
