_base_ = ['./detection_tensorrt_dynamic-320x320-1344x1344.py']

partition_config = dict(type='single_stage', apply_marks=True)
