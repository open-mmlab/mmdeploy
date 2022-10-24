_base_ = ['./video-recognition_static.py']

onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
            1: 'num_crops * num_segs',
            3: 'height',
            4: 'width'
        },
        'output': {
            0: 'batch',
        }
    },
    input_shape=None)
