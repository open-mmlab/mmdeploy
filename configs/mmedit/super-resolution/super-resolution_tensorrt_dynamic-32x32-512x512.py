_base_ = ['./super-resolution_dynamic.py', '../../_base_/backends/tensorrt.py']
backend_config = dict(model_inputs=[
    dict(
        input_shapes=dict(
            input=dict(
                min_shape=[1, 3, 32, 32],
                opt_shape=[1, 3, 256, 256],
                max_shape=[1, 3, 512, 512])))
])
