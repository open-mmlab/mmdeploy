_base_ = ['./segmentation_static.py', '../_base_/backends/ppl.py']

onnx_config = dict(input_shape=None)
