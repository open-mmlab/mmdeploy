_base_ = ['./base_dynamic.py', '../_base_/backends/tensorrt.py']
tensorrt_params = dict(model_params=[
    dict(
        opt_shape_dict=dict(
            input=[[1, 3, 512, 512], [1, 3, 1024, 2048], [1, 3, 2048, 2048]]),
        max_workspace_size=1 << 30)
])
