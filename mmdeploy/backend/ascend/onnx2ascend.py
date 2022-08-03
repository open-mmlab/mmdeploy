# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from subprocess import call

import onnx

from mmdeploy.utils import get_root_logger


def make_shape_string(name, dims):
    return f'{name}:{",".join(map(str, dims))}'


def _concat(dims):
    return ';'.join([','.join(map(str, x)) for x in dims])


def from_onnx(onnx_model, work_dir, model_inputs):
    logger = get_root_logger()
    if not isinstance(onnx_model, str):
        onnx_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
        onnx.save(onnx_model, onnx_path)
    else:
        onnx_path = onnx_model

    onnx_model = onnx.load(onnx_path)
    for n in onnx_model.graph.node:
        if n.domain != '':
            n.domain = ''
    for i in range(1, len(onnx_model.opset_import)):
        onnx_model.opset_import.pop(i)
    onnx.save(onnx_model, onnx_path)

    output_path = osp.join(work_dir, osp.splitext(osp.split(onnx_path)[1])[0])

    input_shapes = []

    for name, dims in model_inputs.input_shapes.items():
        input_shapes.append(make_shape_string(name, dims))
    input_shapes = ','.join(input_shapes)

    input_format = 'ND' if 'dynamic_dims' in model_inputs else 'NCHW'

    args = [
        f'--model={onnx_path}', '--framework=5', f'--output={output_path}',
        '--soc_version=Ascend310', f'--input_format={input_format}',
        f'--input_shape={input_shapes}'
    ]

    if 'dynamic_batch_size' in model_inputs:
        dynamic_batch_size = ",".join(
            map(str, model_inputs.dynamic_batch_size))
        args.append(f'--dynamic_batch_size={dynamic_batch_size}')
    elif 'dynamic_image_size' in model_inputs:
        args.append(
            f'--dynamic_image_size={_concat(model_inputs.dynamic_image_size)}')
    elif 'dynamic_dims' in model_inputs:
        args.append(f'--dynamic_dims={_concat(model_inputs.dynamic_dims)}')

    logger.info(' '.join(('atc', *args)))

    ret_code = call(['atc', *args])
    assert ret_code == 0
