_base_ = ['./base.py']
pytorch2onnx = dict(
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
        },
    }, )
