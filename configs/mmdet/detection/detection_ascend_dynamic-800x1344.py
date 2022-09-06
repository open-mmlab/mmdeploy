_base_ = ['../_base_/base_dynamic.py', '../../_base_/backends/ascend.py']

onnx_config = dict(input_shape=[1344, 800])
backend_config = dict(model_inputs=[
    dict(
        dynamic_image_size=[(800, 1344), (1344, 800)],
        input_shapes=dict(input=[1, 3, -1, -1]))
])
