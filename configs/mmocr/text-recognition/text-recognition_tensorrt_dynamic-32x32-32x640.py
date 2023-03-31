# 3 channel and 32 height input for SATRN models
_base_ = ['./text-recognition_dynamic.py', '../../_base_/backends/tensorrt.py']
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 32, 32],
                    opt_shape=[1, 3, 32, 64],
                    max_shape=[1, 3, 32, 640])))
    ])
