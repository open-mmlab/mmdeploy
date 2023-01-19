_base_ = [
    '../../_base_/torchscript_config.py', '../../_base_/backends/coreml.py'
]

codebase_config = dict(type='mmocr', task='TextRecognition')
backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 32, 32],
                max_shape=[1, 3, 32, 640],
                default_shape=[1, 3, 32, 64])))
])
