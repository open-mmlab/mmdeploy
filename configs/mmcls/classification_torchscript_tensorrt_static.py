_base_ = [
    './classification_torchscript_static.py', '../_base_/backends/tensorrt.py'
]

ir_config = dict(input_shape=None)
