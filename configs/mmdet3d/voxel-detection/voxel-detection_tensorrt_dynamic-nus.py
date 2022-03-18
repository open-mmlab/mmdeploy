_base_ = ['./voxel-detection_dynamic.py', '../../_base_/backends/tensorrt.py']
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                voxels=dict(
                    min_shape=[19760, 20, 5],
                    opt_shape=[19760, 20, 5],
                    max_shape=[19760, 20, 5]),
                num_points=dict(
                    min_shape=[19760], opt_shape=[19760], max_shape=[19760]),
                coors=dict(
                    min_shape=[19760, 4],
                    opt_shape=[19760, 4],
                    max_shape=[19760, 4]),
            ))
    ])
