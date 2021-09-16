_base_ = ['./two-stage_onnxruntime_dynamic.py']

partition_config = dict(type='two_stage', apply_marks=True)
