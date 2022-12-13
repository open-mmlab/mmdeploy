# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, Optional, Union

import onnx
from tvm.relay.frontend import from_onnx as relay_from_onnx
from tvm.relay.quantize import QConfig
from tvm.relay.quantize import qconfig as create_qconfig
from tvm.relay.quantize import quantize
from tvm.target import Target

from mmdeploy.utils import get_root_logger
from .tuner import TVMTunerBase, build_tvm_tuner


def from_onnx(onnx_model: Union[str, onnx.ModelProto],
              output_file: str,
              use_vm: bool = False,
              bytecode_file: str = '',
              shape: Optional[Dict] = None,
              dtype: Union[str, Dict] = 'float32',
              tuner: Optional[Union[TVMTunerBase, Dict]] = None,
              qconfig: Optional[Union[QConfig, Dict]] = None,
              dataset: Optional[Callable] = None):
    """Convert ONNX model to tvm lib.

    Args:
        onnx_model (Union[str, onnx.ModelProto]): ONNX model or model path
        output_file (str): output library path
        use_vm (bool, optional): Enable tvm virtual machine runtime.
            Defaults to False.
        bytecode_file (str, optional): output bytecode path for virtual
            machine. Defaults to ''.
        shape (Optional[Dict], optional): The input shape directory. Defaults
            to None.
        dtype (Union[str, Dict], optional): The input data type dictionary.
            Defaults to 'float32'.
        tuner (Optional[Union[TVMTunerBase, Dict]], optional): The tuner
            config. Defaults to None.

    Return:
        lib: The converted tvm lib
        bytecode: The bytecode of virtual machine runtime.
            None if use_vm==False.

    Examples:
        >>> from mmdeploy.backend.tvm import from_onnx
        >>> onnx_path = 'model.onnx'
        >>> output_file = 'model.so'
        >>> shape = {'input':[1,3,224,224]}
        >>> dtype = {'input':'float32'}
        >>> from_onnx(onnx_path, output_file, shape=shape, dtype=dtype)
    """
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

    # quantization
    if qconfig is not None:
        logger.info('Quantization')

        if isinstance(qconfig, Dict):
            qconfig = create_qconfig(**qconfig)

        with qconfig:
            mod = quantize(mod, params, dataset)

    if tuner is None:
        # use default tuner
        tuner = dict(type='DefaultTuner', target=Target('llvm'))

    if not issubclass(type(tuner), TVMTunerBase):
        tuner['use_vm'] = use_vm
        tuner = build_tvm_tuner(tuner)

    logger.info(f'Tuning with {type(tuner).__name__} .')
    tuner.tune(mod, params)
    lib = tuner.build(mod, params)

    logger.info(f'Export library to {output_file} .')
    bytecode = None
    if tuner.use_vm:
        bytecode, lib = lib.save()
        with open(bytecode_file, mode='wb') as f:
            f.write(bytecode)
    lib.export_library(output_file)
    return lib, bytecode
