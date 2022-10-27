_base_ = ['./pose-detection_static.py', '../_base_/backends/sdk.py']

codebase_config = dict(model_type='sdk')
onnx_config = dict(output_names=['simcc_x', 'simcc_y'])

backend_config = dict(pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='GetBBoxCenterScale'),
    dict(type='PackPoseInputs')
])

ext_info = dict(image_size=[192, 256], padding=1.25)
