_base_ = ['./segmentation_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(input_shape=[320, 320])

codebase_config = dict(model_type='rknn')

backend_config = dict(input_size_list=[[3, 320, 320]])
