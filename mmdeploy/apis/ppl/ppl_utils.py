import logging
import sys

import pyppl.common as pplcommon
import pyppl.nn as pplnn


def register_engines(device_id: int,
                     disable_avx512: bool = False,
                     quick_select: bool = False):
    """Register engines for ppl runtime.

    Args:
        device_id (int): -1 for cpu.
        disable_avx512 (bool): Wheather to disable avx512 for x86.
        quick_select (bool): Wheather to use default algorithms.
    """
    engines = []
    if device_id == -1:
        x86_engine = pplnn.X86EngineFactory.Create()
        if not x86_engine:
            logging.error('Failed to create x86 engine')
            sys.exit(-1)

        if disable_avx512:
            status = x86_engine.Configure(pplnn.X86_CONF_DISABLE_AVX512)
            if status != pplcommon.RC_SUCCESS:
                logging.error('x86 engine Configure() failed: ' +
                              pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

        engines.append(pplnn.Engine(x86_engine))

    else:
        cuda_options = pplnn.CudaEngineOptions()
        cuda_options.device_id = device_id

        cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
        if not cuda_engine:
            logging.error('Failed to create cuda engine.')
            sys.exit(-1)

        if quick_select:
            status = cuda_engine.Configure(
                pplnn.CUDA_CONF_USE_DEFAULT_ALGORITHMS)
            if status != pplcommon.RC_SUCCESS:
                logging.error('cuda engine Configure() failed: ' +
                              pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

        engines.append(pplnn.Engine(cuda_engine))

    return engines
