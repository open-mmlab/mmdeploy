_base_ = ['./two-stage_onnxruntime_static.py']

partition_config = dict(type='two_stage', apply_marks=True)
