_base_ = ['./classification_static.py', '../_base_/backends/ncnn-int8.py']

onnx_config = dict(input_shape=None)
