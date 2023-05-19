_base_ = ['./super-resolution_static.py']
onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'output': {
            0: 'batch',
            2: 'height',
            3: 'width'
        }
    },
    input_shape=None)
