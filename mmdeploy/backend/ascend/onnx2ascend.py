# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from subprocess import call

import onnx


def make_shape_string(name, dims):
    return f'{name}:{",".join(map(str, dims))}'


def from_onnx(onnx_model, work_dir, model_inputs):
    if not isinstance(onnx_model, str):
        onnx_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
        onnx.save(onnx_model, onnx_path)
    else:
        onnx_path = onnx_model

    output_path = osp.join(work_dir, osp.splitext(osp.split(onnx_path)[1])[0])

    print(model_inputs)
    input_shapes = []

    for name, dims in model_inputs.input_shapes.items():
        input_shapes.append(make_shape_string(name, dims))
    input_shapes = ','.join(input_shapes)

    args = [
        f'--model={onnx_path}', '--framework=5', f'--output={output_path}',
        '--soc_version=Ascend310', '--input_format=NCHW',
        f'--input_shape={input_shapes}'
    ]

    if model_inputs.type == 'Static':
        pass
    elif model_inputs.type == 'DynamicBatchSize':
        args.append(
            f'--dynamic_batch_size={",".join(map(str, model_inputs.batch_sizes))}'
        )
    elif model_inputs.type == 'DynamicImageSize':
        sizes = [','.join(map(str, x)) for x in model_inputs.image_sizes]
        args.append(f'--dynamic_image_size={";".join(sizes)}')

    print(' '.join(('atc', *args)))

    ret_code = call(['atc', *args])
    assert ret_code == 0
