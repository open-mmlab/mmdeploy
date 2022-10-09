# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import onnx
from tvm.relay.frontend import from_onnx as relay_from_onnx
from tvm.target import Target

from mmdeploy.utils import get_root_logger
from .tuner import TVMTunerBase, build_tvm_auto_tuner


def from_onnx(onnx_model: Union[str, onnx.ModelProto],
              output_file: str,
              shape: Optional[Dict] = None,
              dtype: Union[str, Dict] = 'float32',
              tuner: Optional[Union[TVMTunerBase, Dict]] = None):
    logger = get_root_logger()

    if shape is not None and isinstance(dtype, Dict):
        assert len(shape) == len(dtype)
        for name in shape:
            assert name in dtype

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)
    assert isinstance(onnx_model, onnx.ModelProto
                      ), f'Expect onnx.ModelProto, but get {type(onnx_model)}.'

    logger.info('Convert onnx to IRModule.')
    mod, params = relay_from_onnx(onnx_model, shape, dtype=dtype, opset=11)

    if tuner is None:
        tuner = dict(type='DefaultTuner', target=Target('llvm'))

    if not issubclass(type(tuner), TVMTunerBase):
        tuner = build_tvm_auto_tuner(tuner)

    logger.info(f'Tuning with {type(tuner).__name__} .')
    tuner.tune(mod, params)
    lib = tuner.build(mod, params)

    logger.info(f'Export library to {output_file} .')
    lib.export_library(output_file)
    return lib
