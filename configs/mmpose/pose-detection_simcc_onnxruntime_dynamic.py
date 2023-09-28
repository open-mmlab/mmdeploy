_base_ = ['./pose-detection_static.py', '../_base_/backends/onnxruntime.py']

onnx_config = dict(
    input_shape=[192, 256],
    output_names=['simcc_x', 'simcc_y'],
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'simcc_x': {
            0: 'batch'
        },
        'simcc_y': {
            0: 'batch'
        }
    })

codebase_config = dict(
    export_postprocess=False  # do not export get_simcc_maximum
)
