_base_ = [
    './text-recognition_static.py', '../../_base_/backends/tensorrt-fp16.py'
]

onnx_config = dict(input_shape=[32, 32])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 1, 32, 32],
                    opt_shape=[1, 1, 32, 32],
                    max_shape=[1, 1, 32, 32])))
    ])
