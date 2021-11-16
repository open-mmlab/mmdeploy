_base_ = ['../_base_/mask_base_dynamic.py', '../../_base_/backends/ppl.py']

onnx_config = dict(input_shape=(1344, 800))

backend_config = dict(model_inputs=dict(opt_shape=[1, 3, 800, 1344]))
