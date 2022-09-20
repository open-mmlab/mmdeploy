_base_ = ['./pose-detection_static.py', '../_base_/backends/sdk.py']

codebase_config = dict(model_type='sdk')

backend_config = dict(pipeline=[
    dict(type='LoadImageFromFile', channel_order='bgr'),
    dict(
        type='PackPoseInputs',
        keys=['img'],
        meta_keys=[
            'id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'input_size',
            'flip_indices', 'category'
        ])
])

ext_info = dict(image_size=[192, 256], padding=1.25)
