_base_ = ['./segmentation_static.py', '../_base_/backends/vacc.py']

backend_config = dict(model_inputs=[dict(shape=dict(input=[1, 3, 512, 512]))])

partition_config = dict(
    type='end2end',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='model.onnx',
            start=['segmentor_forward:output'],
            end=['seg_maps:input'],
            output_names=['feat'])
    ])
