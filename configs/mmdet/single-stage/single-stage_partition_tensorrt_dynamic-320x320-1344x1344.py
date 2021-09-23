_base_ = ['./single-stage_tensorrt_dynamic-320x320-1344x1344.py']

partition_config = dict(type='single_stage', apply_marks=True)
