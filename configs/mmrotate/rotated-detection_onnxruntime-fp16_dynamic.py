_base_ = [
    './rotated-detection_static.py', '../_base_/backends/onnxruntime-fp16.py'
]

onnx_config = dict(
    output_names=['dets', 'labels'],
    input_shape=[1024, 1024],
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
    })

backend_config = dict(
    common_config=dict(op_block_list=['NMSRotated', 'Resize']))
