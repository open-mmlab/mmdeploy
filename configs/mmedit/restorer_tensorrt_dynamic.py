_base_ = ['./base_dynamic.py', '../_base_/backends/tensorrt.py']
tensorrt_params = dict(model_params=[
    dict(
        opt_shape_dict=dict(
            input=[[1, 3, 32, 32], [1, 3, 32, 32], [1, 3, 512, 512]]),
        max_workspace_size=1 << 30)
])
