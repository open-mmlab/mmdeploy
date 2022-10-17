_base_ = ['./text-detection_static.py', '../../_base_/backends/ncnn-int8.py']

onnx_config = dict(input_shape=None)
