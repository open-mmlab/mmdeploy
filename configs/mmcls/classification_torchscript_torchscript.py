_base_ = [
    './classification_torchscript.py', '../_base_/backends/torchscript.py'
]

ir_config = dict(input_shape=None)
