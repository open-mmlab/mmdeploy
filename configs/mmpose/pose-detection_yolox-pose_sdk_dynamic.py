_base_ = ['./pose-detection_static.py', '../_base_/backends/sdk.py']

codebase_config = dict(model_type='sdk_yoloxpose')

backend_config = dict(pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='PoseToDetConverter'),
    dict(type='PackDetPoseInputs')
])
