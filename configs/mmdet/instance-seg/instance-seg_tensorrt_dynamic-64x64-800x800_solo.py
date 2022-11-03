_base_ = [
    '../_base_/base_instance-seg_dynamic_solo.py',
    '../../_base_/backends/tensorrt.py'
]

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 32),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 64, 64],
                    opt_shape=[1, 3, 800, 800],
                    max_shape=[1, 3, 800, 800])))
    ])
