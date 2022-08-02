_base_ = ['./voxel-detection_dynamic.py', '../../_base_/backends/openvino.py']

onnx_config = dict(input_shape=None)

backend_config = dict(model_inputs=[
    dict(
        opt_shapes=dict(
            voxels=[5000, 32, 4], num_points=[5000], coors=[5000, 4]))
])
