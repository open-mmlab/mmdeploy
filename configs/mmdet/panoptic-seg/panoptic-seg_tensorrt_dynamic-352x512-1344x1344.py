_base_ = [
    '../_base_/base_panoptic-seg_dynamic.py',
    '../../_base_/backends/tensorrt.py'
]

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 352, 512],
                    opt_shape=[1, 3, 800, 1344],
                    max_shape=[1, 3, 1344, 1344])))
    ])
