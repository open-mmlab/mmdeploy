_base_ = ['./single-stage_tensorrt_dynamic-300x300-512x512.py']

partition_config = dict(type='single_stage', apply_marks=True)
