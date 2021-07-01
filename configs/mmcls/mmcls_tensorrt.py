_base_ = ['./mmcls_base.py', '../_base_/backends/tensorrt.py']
tensorrt_param = dict(model_params=[
    dict(
        save_file='end2end.engine',
        opt_shape_dict=dict(
            input=[[1, 3, 224, 224], [4, 3, 224, 224], [32, 3, 224, 224]]),
        max_workspace_size=1 << 30)
])
