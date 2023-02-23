_base_ = ['./segmentation_static.py', '../_base_/backends/vacc.py']

onnx_config = dict(input_shape=[512, 512])

backend_config = dict(
    common_config=dict(
        vdsp_params_info=dict(
            vdsp_op_type=301,
            iimage_format=5000,
            iimage_width=512,
            iimage_height=512,
            oimage_width=512,
            oimage_height=512,
            iimage_width_pitch=512,
            iimage_height_pitch=512,
            resize_type=1,
            color_cvt_code=2,
            color_space=0,
            meanr=22459,
            meang=22340,
            meanb=22136,
            stdr=21325,
            stdg=21284,
            stdb=21292,
            norma_type=3)),
    model_inputs=[dict(shape=dict(input=[1, 3, 512, 512]))])

codebase_config = dict(model_type='vacc_seg')

partition_config = dict(
    type='vacc_seg',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='model.onnx',
            start=['segmentor_forward:output'],
            # 'decode_head' will skip `ArgMax`
            # 'seg_maps' will skip `Resize` and `ArgMax`
            end=['decode_head:input'],
            output_names=['feat'])
    ])
