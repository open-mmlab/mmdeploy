_base_ = ['./voxel-detection_dynamic.py', '../../_base_/backends/tensorrt.py']
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                voxels=dict(
                    min_shape=[3000, 32, 4],
                    opt_shape=[4000, 32, 4],
                    max_shape=[5000, 32, 4]),
                num_points=dict(
                    min_shape=[3000], opt_shape=[4000], max_shape=[5000]),
                coors=dict(
                    min_shape=[3000, 4],
                    opt_shape=[4000, 4],
                    max_shape=[5000, 4]),
            ))
    ])
