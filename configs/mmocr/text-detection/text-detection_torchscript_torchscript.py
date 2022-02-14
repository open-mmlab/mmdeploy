_base_ = [
    './text-detection_torchscript.py', '../../_base_/backends/torchscript.py'
]

ir_config = dict(input_shape=None)
