_base_ = ['./base.py', '../_base_/backends/tensorrt.py']
tensorrt_params = dict(model_params=[
    dict(
        opt_shape_dict=dict(
            input=[[1, 3, 512, 1024], [1, 3, 512, 1024], [1, 3, 512, 1024]]),
        max_workspace_size=1 << 30)
])
