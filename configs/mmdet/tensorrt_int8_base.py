_base_ = ['../_base_/backends/tensorrt_int8.py']
tensorrt_params = dict(model_params=[
    dict(
        opt_shape_dict=dict(
            input=[[1, 3, 320, 320], [1, 3, 800, 1344], [1, 3, 1344, 1344]]),
        max_workspace_size=1 << 30)
])
