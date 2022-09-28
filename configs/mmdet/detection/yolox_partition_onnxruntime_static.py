_base_ = ['./detection_onnxruntime_static.py']

onnx_config = dict(input_shape=[640, 640])
partition_config = dict(
    type='yolox_partition',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='yolox.onnx',
            start=['detector_forward:input'],
            end=['yolox_head:input'])
    ])
