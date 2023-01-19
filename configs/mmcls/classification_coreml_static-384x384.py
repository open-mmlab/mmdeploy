_base_ = ['./classification_coreml_dynamic-224x224-224x224.py']

ir_config = dict(input_shape=(384, 384))
backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 384, 384],
                max_shape=[1, 3, 384, 384],
                default_shape=[1, 3, 384, 384])))
])
