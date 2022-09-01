# Copyright (c) OpenMMLab. All rights reserved.
import sys
from typing import List, Sequence

import pyppl.common as pplcommon
import pyppl.nn as pplnn

from mmdeploy.utils import get_root_logger


def create_runtime(onnx_file: str,
                   engines: List[pplnn.Engine]) -> pplnn.Runtime:
    """Create runtime object for pplnn.

    Args:
        onnx_file (str): path to onnx model
        engines (List[pplnn.Engine]): engines used to create the runtime
            object

    Returns:
        pplnn.Runtime: created runtime object
    """
    runtime_builder = pplnn.onnx.RuntimeBuilderFactory.Create()
    assert runtime_builder is not None, 'Failed to create '\
        'onnx.RuntimeBuilder.'

    status = runtime_builder.LoadModelFromFile(onnx_file)
    assert status == pplcommon.RC_SUCCESS, 'Failed to load ONNX model.'

    resources = pplnn.onnx.RuntimeBuilderResources()
    resources.engines = engines

    status = runtime_builder.SetResources(resources)
    assert status == pplcommon.RC_SUCCESS, 'runtime_builder.SetResources() ' \
        'Failed.'

    status = runtime_builder.Preprocess()
    assert status == pplcommon.RC_SUCCESS, 'runtime_builder.Preprocess() ' \
        'Failed.'

    runtime = runtime_builder.CreateRuntime()
    assert runtime is not None, 'Failed to create onnx.Runtime'
    return runtime


def register_engines(device_id: int,
                     disable_avx512: bool = False,
                     quick_select: bool = False,
                     input_shapes: Sequence[Sequence[int]] = None,
                     export_algo_file: str = None,
                     import_algo_file: str = None) -> List[pplnn.Engine]:
    """Register engines for pplnn runtime.

    Args:
        device_id (int): Specifying device index. `-1` for cpu.
        disable_avx512 (bool): Whether to disable avx512 for x86.
            Defaults to `False`.
        quick_select (bool): Whether to use default algorithms.
            Defaults to `False`.
        input_shapes (Sequence[Sequence[int]]): shapes for PPLNN optimization.
        export_algo_file (str): File path for exporting PPLNN optimization
            file.
        import_algo_file (str): File path for loading PPLNN optimization file.
    Returns:
        list[pplnn.Engine]: A list of registered pplnn engines.
    """
    engines = []
    logger = get_root_logger()
    if device_id == -1:
        x86_options = pplnn.x86.EngineOptions()
        x86_engine = pplnn.x86.EngineFactory.Create(x86_options)
        if not x86_engine:
            logger.error('Failed to create x86 engine')
            sys.exit(1)

        if disable_avx512:
            status = x86_engine.Configure(pplnn.x86.ENGINE_CONF_DISABLE_AVX512)
            if status != pplcommon.RC_SUCCESS:
                logger.error('x86 engine Configure() failed: ' +
                             pplcommon.GetRetCodeStr(status))
                sys.exit(1)

        engines.append(x86_engine)

    else:
        cuda_options = pplnn.cuda.EngineOptions()
        cuda_options.device_id = device_id
        cuda_options.mm_policy = pplnn.cuda.MM_BEST_FIT

        cuda_engine = pplnn.cuda.EngineFactory.Create(cuda_options)
        if not cuda_engine:
            logger.error('Failed to create cuda engine.')
            sys.exit(1)

        if quick_select:
            status = cuda_engine.Configure(
                pplnn.cuda.ENGINE_CONF_USE_DEFAULT_ALGORITHMS)
            if status != pplcommon.RC_SUCCESS:
                logger.error('cuda engine Configure() failed: ' +
                             pplcommon.GetRetCodeStr(status))
                sys.exit(1)

        if input_shapes is not None:
            status = cuda_engine.Configure(
                pplnn.cuda.ENGINE_CONF_SET_INPUT_DIMS, input_shapes)
            if status != pplcommon.RC_SUCCESS:
                logger.error(
                    'cuda engine Configure(ENGINE_CONF_SET_INPUT_DIMS) '
                    'failed: ' + pplcommon.GetRetCodeStr(status))
                sys.exit(1)

        if export_algo_file is not None:
            status = cuda_engine.Configure(
                pplnn.cuda.ENGINE_CONF_EXPORT_ALGORITHMS, export_algo_file)
            if status != pplcommon.RC_SUCCESS:
                logger.error(
                    'cuda engine Configure(ENGINE_CONF_EXPORT_ALGORITHMS) '
                    'failed: ' + pplcommon.GetRetCodeStr(status))
                sys.exit(1)

        if import_algo_file is not None:
            status = cuda_engine.Configure(
                pplnn.cuda.ENGINE_CONF_IMPORT_ALGORITHMS, import_algo_file)
            if status != pplcommon.RC_SUCCESS:
                logger.error(
                    'cuda engine Configure(ENGINE_CONF_IMPORT_ALGORITHMS) '
                    'failed: ' + pplcommon.GetRetCodeStr(status))
                sys.exit(1)

        engines.append(cuda_engine)

    return engines
