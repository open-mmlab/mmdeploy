_base_ = ['./monocular-detection_static.py']

onnx_config = dict(
    dynamic_axes={
        'img': {
            2: 'height',
            3: 'width',
        },
        'bboxes': {
            1: 'num_dets',
        },
        'scores': {
            1: 'num_dets'
        },
        'labels': {
            1: 'num_dets'
        },
        'dir_scores': {
            1: 'num_dets'
        },
        'attrs': {
            1: 'num_dets'
        }
    }, )
