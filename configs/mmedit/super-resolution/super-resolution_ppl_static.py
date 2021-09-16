_base_ = ['./super-resolution_static.py', '../../_base_/backends/ppl.py']

onnx_config = dict(input_shape=[256, 256])
