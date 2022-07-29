_base_ = ['../_base_/base_static.py', '../../_base_/backends/rknn.py']

backend_config = dict(
    common_config=dict(mean_values=[0, 0, 0], std_values=[255., 255., 255.]),
    input_size_list=[[3, 608, 416]])

partition_config = dict(
    type='single_stage',  # the partition policy name
    apply_marks=True,  # should always be set to True
    partition_cfg=[
        dict(
            save_file=
            'yolov3.onnx',  # filename to save the partitioned onnx model
            start=['detector_forward:input'],  # [mark_name:input/output, ...]
            end=['yolo_head:input'])  # [mark_name:input/output, ...]
    ])
