_base_ = [
    './monocular-detection_static.py', '../../_base_/backends/tensorrt.py'
]

onnx_config = dict(input_shape=(1600, 928))

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 928, 1600],
                    opt_shape=[1, 3, 928, 1600],
                    max_shape=[1, 3, 928, 1600])))
    ])
