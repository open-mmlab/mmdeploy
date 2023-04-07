_base_ = ['./inpainting_static.py']

onnx_config = dict(
    dynamic_axes=dict(
        masked_img={
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        mask={
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        output={
            0: 'batch',
            2: 'height',
            3: 'width'
        }),
    input_shape=None)
