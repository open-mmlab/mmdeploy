_base_ = ['./detection_tensorrt_dynamic-320x320-1344x1344.py']

partition_config = dict(type='two_stage', apply_marks=True)
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 800, 1344],
                    max_shape=[1, 3, 1344, 1344]))),
        dict(
            input_shapes=dict(
                bbox_feats=dict(
                    min_shape=[500, 256, 7, 7],
                    opt_shape=[1000, 256, 7, 7],
                    max_shape=[2000, 256, 7, 7])))
    ])
