_base_ = [
    './text-detection_dynamic.py', '../../_base_/backends/tensorrt-fp16.py'
]
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 600, 800],
                    max_shape=[1, 3, 2240, 2240])))
    ])
