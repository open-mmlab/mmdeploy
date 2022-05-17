# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmdeploy.backend.tensorrt import create_trt_engine, save_trt_engine
from mmdeploy.backend.tensorrt.utils import get_trt_log_level


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT.')
    parser.add_argument('onnx_path', help='ONNX model path')
    parser.add_argument('output', help='output TensorRT engine path')
    args = parser.parse_args()

    return args


def main():
    # This only for test mmdeploy convert to trt engine.
    # The command  to convert the backend.
    #
    # python tools/deploy.py configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \ # noqa
    #        ../mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    #        ~/driver_for_mmdeploy/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth \  # noqa
    #        ../mmdetection/demo/demo.jpg  \
    #        --device cuda  \
    #        --work-dir  \
    #        ~/convert_mmdeploy/retinanet  \
    #        --dump-info
    args = parse_args()

    # All the value below get from: configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py  # noqa
    input_shapes = {
        'input': {
            'min_shape': [1, 3, 320, 320],
            'opt_shape': [1, 3, 800, 1344],
            'max_shape': [1, 3, 1344, 1344]
        }
    }

    engine = create_trt_engine(args.onnx_path,
                               input_shapes=input_shapes,
                               log_level=get_trt_log_level(),
                               fp16_mode=False,
                               int8_mode=False,
                               int8_param={},
                               max_workspace_size=1073741824,
                               device_id=0)

    save_trt_engine(engine, args.output)


if __name__ == '__main__':
    main()
