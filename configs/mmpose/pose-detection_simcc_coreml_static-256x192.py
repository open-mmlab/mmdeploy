_base_ = ['../_base_/torchscript_config.py', '../_base_/backends/coreml.py']

ir_config = dict(input_shape=[256, 192], output_names=['simcc_x', 'simcc_y'])

codebase_config = dict(type='mmpose', task='PoseDetection')

backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 192, 256],
                max_shape=[1, 3, 192, 256],
                default_shape=[1, 3, 192, 256])))
])
