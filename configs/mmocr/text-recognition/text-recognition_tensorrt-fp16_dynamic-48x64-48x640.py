# 3 channel and 48 height for SAR models
_base_ = [
    './text-recognition_dynamic.py', '../../_base_/backends/tensorrt-fp16.py'
]
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 48, 64],
                    opt_shape=[1, 3, 48, 64],
                    max_shape=[1, 3, 48, 640])))
    ])
