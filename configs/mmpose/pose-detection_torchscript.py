_base_ = [
    '../_base_/torchscript_config.py', '../_base_/backends/torchscript.py'
]

codebase_config = dict(type='mmpose', task='PoseDetection')
ir_config = dict(input_shape=[192, 256])
