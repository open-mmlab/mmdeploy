# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from subprocess import call
from typing import Dict, Sequence

from mmdeploy.utils import get_root_logger


def make_shape_string(name, dims):
    return f'{name}:{",".join(map(str, dims))}'


def _concat(dims: Sequence) -> str:
    return ';'.join([','.join(map(str, x)) for x in dims])


@dataclass
class AtcParam:
    input_shapes: Dict[str, Sequence] = field(default_factory=OrderedDict)
    dynamic_batch_size: Sequence[int] = None
    dynamic_image_size: Sequence[Sequence[int]] = None
    dynamic_dims: Sequence[Sequence[int]] = None

    def check(self):
        dynamic_count = 0
        if self.dynamic_batch_size is not None:
            dynamic_count += 1
        if self.dynamic_image_size is not None:
            dynamic_count += 1
        if self.dynamic_dims is not None:
            dynamic_count += 1

        if dynamic_count > 1:
            raise ValueError('Expect one dynamic flag, but got: '
                             f'dynamic_batch_size {self.dynamic_batch_size}; '
                             f'dynamic_image_size {self.dynamic_image_size}; '
                             f'dynamic_dims {self.dynamic_dims}; ')


def from_onnx(onnx_model: str, output_path: str, atc_param: AtcParam):
    """Convert ONNX to Ascend model.

    Example:
        >>> from mmdeploy.backend.ascend.onnx2ascend import AtcParam, from_onnx
        >>> onnx_path = 'work_dir/end2end.onnx'
        >>> output_path = 'work_dir/end2end.om
        >>> atc_param = AtcParam(input_shapes=dict(input=[1, 3, 224, 224]))
        >>> from_onnx(onnx_path, output_path, atc_param)

    Args:
        onnx_path (ModelProto|str): The path of the onnx model.
        output_path (str): Path to save model.
        atc_param (AtcParam): The input args to the atc tools.
    """
    import onnx
    logger = get_root_logger()
    atc_param.check()

    if not isinstance(onnx_model, str):
        onnx_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
        onnx.save(onnx_model, onnx_path)
    else:
        onnx_path = onnx_model

    onnx_model = onnx.load(onnx_path)
    input_names = [i.name for i in onnx_model.graph.input]
    for n in onnx_model.graph.node:
        if n.domain != '':
            n.domain = ''
    for i in range(1, len(onnx_model.opset_import)):
        onnx_model.opset_import.pop(i)
    onnx.save(onnx_model, onnx_path)

    input_shapes = []

    for name, dims in atc_param.input_shapes.items():
        input_shapes.append(make_shape_string(name, dims))
    input_shapes = ';'.join(input_shapes)

    input_format = 'ND' if atc_param.dynamic_dims is not None else 'NCHW'

    if output_path.endswith('.om'):
        output_path = osp.splitext(output_path)[0]

    args = [
        f'--model={onnx_path}', '--framework=5', f'--output={output_path}',
        '--soc_version=Ascend310', f'--input_format={input_format}',
        f'--input_shape={input_shapes}'
    ]

    if atc_param.dynamic_batch_size is not None:
        dynamic_batch_size = ','.join(map(str, atc_param.dynamic_batch_size))
        args.append(f'--dynamic_batch_size={dynamic_batch_size}')
    elif atc_param.dynamic_image_size is not None:
        dynamic_image_size = atc_param.dynamic_image_size
        if isinstance(dynamic_image_size, Dict):
            dynamic_image_size = [
                dynamic_batch_size[name] for name in input_names
                if name in dynamic_batch_size
            ]
        dynamic_image_size = _concat(dynamic_image_size)
        args.append(f'--dynamic_image_size={dynamic_image_size}')
    elif atc_param.dynamic_dims is not None:
        dynamic_dims = atc_param.dynamic_dims
        if isinstance(dynamic_dims, Dict):
            dynamic_dims = [
                dynamic_dims[name] for name in input_names
                if name in dynamic_dims
            ]
        dynamic_dims = _concat(dynamic_dims)
        args.append(f'--dynamic_dims={dynamic_dims}')

    logger.info(' '.join(('atc', *args)))

    ret_code = call(['atc', *args])
    assert ret_code == 0
