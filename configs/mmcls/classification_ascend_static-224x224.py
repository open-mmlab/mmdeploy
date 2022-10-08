_base_ = ['./classification_static.py', '../_base_/backends/ascend.py']

onnx_config = dict(input_shape=[224, 224])
backend_config = dict(
    model_inputs=[dict(input_shapes=dict(input=[1, 3, 224, 224]))])
