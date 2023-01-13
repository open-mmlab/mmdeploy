_base_ = ['../_base_/base_torchscript.py', '../../_base_/backends/coreml.py']

ir_config = dict(input_shape=(608, 608))
backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 608, 608],
                max_shape=[1, 3, 608, 608],
                default_shape=[1, 3, 608, 608])))
])
