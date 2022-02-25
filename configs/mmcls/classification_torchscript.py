_base_ = [
    '../_base_/torchscript_config.py', '../_base_/backends/torchscript.py'
]

ir_config = dict(input_shape=None)
codebase_config = dict(type='mmcls', task='Classification')
