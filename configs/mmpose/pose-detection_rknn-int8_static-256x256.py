_base_ = ['./pose-detection_static.py', '../_base_/backends/rknn.py']

onnx_config = dict(input_shape=[256, 256])

codebase_config = dict(model_type='end2end')

backend_config = dict(
    input_size_list=[[3, 256, 256]],
    common_config=dict(target_platform='rv1126'))
