_base_ = ['./pose-detection_static.py', '../_base_/backends/sdk.py']

codebase_config = dict(model_type='sdk')

backend_config = dict(pipeline=[
    dict(type='LoadImageFromFile', channel_order='bgr'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'bbox', 'rotation', 'bbox_score', 'flip_pairs'
        ])
])

ext_info = dict(image_size=[192, 256], padding=1.25)
