_base_ = ['./text-detection_dynamic.py', '../../_base_/backends/ascend.py']

onnx_config = dict(input_shape=None)
backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(input=[-1, 3, -1, -1]),
        dynamic_dims=[(1, 640, 640), (4, 640, 640), (1, 1280, 1280)])
])
