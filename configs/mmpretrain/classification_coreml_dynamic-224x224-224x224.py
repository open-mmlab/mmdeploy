_base_ = ['../_base_/torchscript_config.py', '../_base_/backends/coreml.py']

codebase_config = dict(type='mmpretrain', task='Classification')

backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 224, 224],
                max_shape=[8, 3, 224, 224],
                default_shape=[1, 3, 224, 224])))
])
