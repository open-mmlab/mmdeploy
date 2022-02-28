_base_ = ['./pose-detection_static.py', '../_base_/backends/pplnn.py']

onnx_config = dict(input_shape=[192, 256])

backend_config = dict(model_inputs=dict(opt_shape=[1, 3, 256, 192]))
