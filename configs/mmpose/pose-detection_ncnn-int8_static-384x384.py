_base_ = ['./pose-detection_static.py', '../_base_/backends/ncnn-int8.py']

onnx_config = dict(input_shape=[384, 384])
