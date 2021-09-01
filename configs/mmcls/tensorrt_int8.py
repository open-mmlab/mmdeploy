_base_ = ['./base_dynamic.py', '../_base_/backends/tensorrt_int8.py']
tensorrt_params = dict(model_params=[
    dict(
        save_file='end2end.engine',
        opt_shape_dict=dict(
            input=[[1, 3, 224, 224], [1, 3, 224, 224], [64, 3, 224, 224]]),
        max_workspace_size=1 << 30)
])
