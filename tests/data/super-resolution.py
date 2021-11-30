# Copyright (c) OpenMMLab. All rights reserved.
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
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=None)

codebase_config = dict(type='mmedit', task='SuperResolution')
backend_config = dict(type='onnxruntime')
