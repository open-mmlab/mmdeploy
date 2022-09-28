_base_ = ['./detection_onnxruntime_static.py']

onnx_config = dict(input_shape=[640, 640])
partition_config = dict(
    type='fast_rcnn_partition',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='fast_rcnn.onnx',
            start=['detector_forward:input'],
            end=['rpn_head:input'])
    ])
