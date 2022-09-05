_base_ = [
    '../_base_/torchscript_config.py', '../_base_/backends/coreml.py',
    './segmentation_static.py'
]

ir_config = dict(input_shape=[1024, 512])
backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 512, 1024],
                max_shape=[1, 3, 512, 1024],
                default_shape=[1, 3, 512, 1024])))
])
