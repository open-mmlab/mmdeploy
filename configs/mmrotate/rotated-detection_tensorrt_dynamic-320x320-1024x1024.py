_base_ = ['./rotated-detection_static.py', '../_base_/backends/tensorrt.py']

onnx_config = dict(
    output_names=['dets', 'labels'],
    input_shape=None,
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'dets': {
            0: 'batch',
            1: 'num_dets',
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
        },
    },
)

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 1024, 1024],
                    max_shape=[1, 3, 1024, 1024])))
    ])
