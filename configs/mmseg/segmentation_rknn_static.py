_base_ = ['./segmentation_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(input_shape=[512, 512])

codebase_config = dict(model_type='rknn')

backend_config = dict(input_size_list=[[3, 512, 512]])
