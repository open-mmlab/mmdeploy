_base_ = ['./instance-seg_onnxruntime_static.py']

onnx_config = dict(input_shape=[640, 640])
partition_config = dict(
    type='mask_rcnn_partition',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='mask_rcnn.onnx',
            start=['detector_forward:input'],
            end=['rpn_head:input'])
    ])
