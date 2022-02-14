_base_ = [
    './text-recognition_torchscript.py', '../../_base_/backends/torchscript.py'
]

ir_config = dict(input_shape=None)
