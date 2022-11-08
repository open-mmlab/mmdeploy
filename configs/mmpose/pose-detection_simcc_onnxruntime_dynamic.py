_base_ = ['./pose-detection_static.py', '../_base_/backends/onnxruntime.py']

onnx_config = dict(
    input_shape=[192, 256],
    output_names=['simcc_x', 'simcc_y'],
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'output': {
            0: 'batch'
        }
    })
