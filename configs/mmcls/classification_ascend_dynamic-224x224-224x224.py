_base_ = ['./classification_dynamic.py', '../_base_/backends/ascend.py']

onnx_config = dict(input_shape=[224, 224])

backend_config = dict(model_inputs=[
    dict(
        dynamic_batch_size=[1, 2, 4, 8],
        input_shapes=dict(input=[-1, 3, 224, 224]))
])
