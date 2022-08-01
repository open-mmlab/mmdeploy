_base_ = ['./classification_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(input_shape=None)
codebase_config = dict(model_type='rknn')
backend_config = dict(input_size_list=[[3, 224, 224]])
