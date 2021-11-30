_base_ = ['../_base_/base_static.py', '../../_base_/backends/ncnn.py']

partition_config = dict(type='single_stage', apply_marks=True)
