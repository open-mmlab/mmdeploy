_base_ = ['./voxel-detection_dynamic.py', '../../_base_/backends/tensorrt.py']
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 32),
    model_inputs=[
        dict(
            input_shapes=dict(
                voxels=dict(
                    min_shape=[5000, 64, 4],
                    opt_shape=[20000, 64, 4],
                    max_shape=[30000, 64, 4]),
                num_points=dict(
                    min_shape=[5000], opt_shape=[20000], max_shape=[30000]),
                coors=dict(
                    min_shape=[5000, 4],
                    opt_shape=[20000, 4],
                    max_shape=[30000, 4]),
            ))
    ])
