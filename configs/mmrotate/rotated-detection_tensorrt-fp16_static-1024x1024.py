_base_ = [
    './rotated-detection_static.py', '../_base_/backends/tensorrt-fp16.py'
]

onnx_config = dict(output_names=['dets', 'labels'], input_shape=(1024, 1024))

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 1024, 1024],
                    opt_shape=[1, 3, 1024, 1024],
                    max_shape=[1, 3, 1024, 1024])))
    ])
