_base_ = ['./voxel-detection_dynamic.py', '../../_base_/backends/tensorrt.py']
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                voxels=dict(
                    min_shape=[2000, 32, 4],
                    opt_shape=[5000, 32, 4],
                    max_shape=[9000, 32, 4]),
                num_points=dict(
                    min_shape=[2000], opt_shape=[5000], max_shape=[9000]),
                coors=dict(
                    min_shape=[2000, 4],
                    opt_shape=[5000, 4],
                    max_shape=[9000, 4]),
            ))
    ])
