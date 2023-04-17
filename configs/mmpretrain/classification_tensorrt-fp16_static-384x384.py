_base_ = ['./classification_static.py', '../_base_/backends/tensorrt-fp16.py']

onnx_config = dict(input_shape=[384, 384])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 384, 384],
                    opt_shape=[1, 3, 384, 384],
                    max_shape=[1, 3, 384, 384])))
    ])
