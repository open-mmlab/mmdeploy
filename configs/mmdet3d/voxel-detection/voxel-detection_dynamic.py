_base_ = ['./voxel-detection_static.py']

onnx_config = dict(
    dynamic_axes={
        'voxels': {
            0: 'voxels_num',
        },
        'num_points': {
            0: 'voxels_num',
        },
        'coors': {
            0: 'voxels_num',
        }
    },
    input_shape=None)
