_base_ = ['./super-resolution_dynamic.py', '../../_base_/backends/ppl.py']

onnx_config = dict(input_shape=(32, 32))

backend_config = dict(model_inputs=dict(opt_shape=[1, 3, 32, 32]))
