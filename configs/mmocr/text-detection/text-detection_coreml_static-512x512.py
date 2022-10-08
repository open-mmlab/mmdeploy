_base_ = [
    './text-detection_static.py', '../../_base_/torchscript_config.py',
    '../../_base_/backends/coreml.py'
]

ir_config = dict(input_shape=[512, 512])
backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 512, 512],
                max_shape=[1, 3, 512, 512],
                default_shape=[1, 3, 512, 512])))
])
