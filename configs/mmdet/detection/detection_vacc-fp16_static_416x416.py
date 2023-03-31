_base_ = ['../_base_/base_static.py', '../../_base_/backends/vacc.py']

onnx_config = dict(input_shape=[416, 416])

backend_config = dict(
    common_config=dict(
        vdsp_params_info=dict(
            vdsp_op_type=303,
            iimage_format=5000,
            iimage_width=640,
            iimage_height=640,
            oimage_width=416,
            oimage_height=416,
            iimage_width_pitch=640,
            iimage_height_pitch=640,
            resize_type=1,
            color_cvt_code=2,
            color_space=0,
            padding_value_r=114,
            padding_value_g=114,
            padding_value_b=114,
            edge_padding_type=0,
            meanr=0,
            meang=0,
            meanb=0,
            stdr=23544,
            stdg=23544,
            stdb=23544,
            norma_type=3)),
    model_inputs=[
        dict(shape=dict(input=[1, 3, 416, 416]), qconfig=dict(dtype='fp16'))
    ])

partition_config = dict(
    type='vacc_det',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='model.onnx',
            start=['detector_forward:input'],
            end=['yolo_head:input'],
            output_names=[f'pred_maps.{i}' for i in range(3)])
    ])
