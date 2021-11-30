_base_ = ['./detection_onnxruntime_static.py']

partition_config = dict(type='single_stage', apply_marks=True)
