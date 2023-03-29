_base_ = ['./inpainting_static.py', '../../_base_/backends/tensorrt-int8.py']

onnx_config = dict(input_shape=[256, 256])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                masked_img=dict(
                    min_shape=[1, 3, 256, 256],
                    opt_shape=[1, 3, 256, 256],
                    max_shape=[1, 3, 256, 256]),
                mask=dict(
                    min_shape=[1, 1, 256, 256],
                    opt_shape=[1, 1, 256, 256],
                    max_shape=[1, 1, 256, 256])))
    ])
