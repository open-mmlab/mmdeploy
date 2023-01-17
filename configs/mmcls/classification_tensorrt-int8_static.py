_base_ = ['./classification_static.py', '../_base_/backends/tensorrt-int8.py']

onnx_config = dict(input_shape=[224, 224])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 224, 224],
                    opt_shape=[1, 3, 224, 224],
                    max_shape=[1, 3, 224, 224])))
    ])
