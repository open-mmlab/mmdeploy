_base_ = ['./segmentation_static.py', '../_base_/backends/vacc.py']

onnx_config = dict(input_shape=[512, 512])

backend_config = dict(model_inputs=[
    dict(shape=dict(input=[1, 3, 512, 512]), qconfig=dict(dtype='int8'))
])

partition_config = dict(
    type='end2end',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='model.onnx',
            start=['segmentor_forward:output'],
            end=['decode_head:input'],
            output_names=['feat'])
    ])
