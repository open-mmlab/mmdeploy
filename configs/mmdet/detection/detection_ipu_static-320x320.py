_base_ = ['../_base_/base_static.py', '../../_base_/backends/ipu.py']

backend_config = dict(input_shape='input=1,3,320,320')

onnx_config = dict(input_shape=[320, 320])
