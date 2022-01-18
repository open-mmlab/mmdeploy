_base_ = [
    './classification_torchscript_static.py',
    '../_base_/backends/torchscript.py'
]

ir_config = dict(input_shape=None)
