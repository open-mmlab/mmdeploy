_base_ = ['./partition_two_stage.py', '../_base_/backends/tensorrt_int8.py']

tensorrt_params = dict(model_params=[
    dict(
        opt_shape_dict=dict(
            input=[[1, 3, 320, 320], [1, 3, 800, 1344], [1, 3, 1344, 1344]]),
        max_workspace_size=1 << 30),
    dict(
        opt_shape_dict=dict(bbox_feats=[[500, 256, 7, 7], [1000, 256, 7, 7],
                                        [2000, 256, 7, 7]]),
        max_workspace_size=1 << 30)
])
