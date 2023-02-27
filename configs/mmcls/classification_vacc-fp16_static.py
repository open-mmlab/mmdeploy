_base_ = ['./classification_static.py', '../_base_/backends/vacc.py']

backend_config = dict(
    common_config=dict(
        vdsp_params_info=dict(
            vdsp_op_type=300,
            iimage_format=5000,
            iimage_width=256,
            iimage_height=256,
            iimage_width_pitch=256,
            iimage_height_pitch=256,
            short_edge_threshold=256,
            resize_type=1,
            color_cvt_code=2,
            color_space=0,
            crop_size=224,
            meanr=22459,
            meang=22340,
            meanb=22136,
            stdr=21325,
            stdg=21284,
            stdb=21292,
            norma_type=3)),
    model_inputs=[
        dict(shape=dict(input=[1, 3, 224, 224]), qconfig=dict(dtype='fp16'))
    ])
