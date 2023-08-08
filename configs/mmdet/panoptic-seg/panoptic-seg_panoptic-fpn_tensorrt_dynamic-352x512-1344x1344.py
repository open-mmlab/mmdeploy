_base_ = [
    '../_base_/base_panoptic-seg_static.py',
    '../../_base_/backends/tensorrt.py'
]
onnx_config = dict(
    input_shape=None,
    output_names=['dets', 'labels', 'masks', 'semseg'],
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'dets': {
            0: 'batch',
            1: 'num_dets',
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
        },
        'masks': {
            0: 'batch',
            1: 'num_dets',
        },
        'semseg': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
    },
)

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 352, 512],
                    opt_shape=[1, 3, 800, 1344],
                    max_shape=[1, 3, 1344, 1344])))
    ])
