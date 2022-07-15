_base_ = ['./segmentation_static.py', '../_base_/backends/snpe.py']

onnx_config = dict(input_shape=[512, 1024])
