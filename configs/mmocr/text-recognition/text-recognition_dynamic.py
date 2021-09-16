_base_ = ['./text-recognition_static.py']
onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
            3: 'width'
        },
        'output': {
            0: 'batch',
            3: 'width'
        }
    }, )
