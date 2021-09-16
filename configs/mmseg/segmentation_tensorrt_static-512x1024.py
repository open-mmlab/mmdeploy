_base_ = ['./segmentation_static.py', '../_base_/backends/tensorrt.py']

onnx_config = dict(input_shape=[1024, 512])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 512, 1024],
                    opt_shape=[1, 3, 512, 1024],
                    max_shape=[1, 3, 512, 1024])))
    ])
