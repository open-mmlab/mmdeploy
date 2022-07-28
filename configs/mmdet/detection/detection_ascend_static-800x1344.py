_base_ = ['../_base_/base_static.py', '../../_base_/backends/ascend.py']

onnx_config = dict(input_shape=[1344, 800])
backend_config = dict(
    model_inputs=[dict(input_shapes=dict(input=[1, 3, 800, 1344]))])
