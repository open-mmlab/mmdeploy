_base_ = ['./segmentation_dynamic.py', '../_base_/backends/pplnn.py']

onnx_config = dict(input_shape=[2048, 1024])

backend_config = dict(model_inputs=dict(opt_shape=[1, 3, 1024, 2048]))
