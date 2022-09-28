_base_ = ['./detection_onnxruntime_static.py']

onnx_config = dict(input_shape=[640, 640])
partition_config = dict(
    type='yolo_partition',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='yolo.onnx',
            start=['detector_forward:input'],
            end=['yolo_head:input'])
    ])
