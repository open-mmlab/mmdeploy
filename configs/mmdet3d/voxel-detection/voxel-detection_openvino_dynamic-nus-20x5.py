_base_ = ['./voxel-detection_dynamic.py', '../../_base_/backends/openvino.py']

onnx_config = dict(input_shape=None)

backend_config = dict(model_inputs=[
    dict(
        opt_shapes=dict(
            voxels=[20000, 20, 5], num_points=[20000], coors=[20000, 4]))
])
