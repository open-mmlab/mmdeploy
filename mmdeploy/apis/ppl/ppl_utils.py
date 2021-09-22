import logging
import sys

import numpy as np
import pyppl.common as pplcommon
import pyppl.nn as pplnn
import torch

from mmdeploy.utils.timer import TimeCounter


def register_engines(device_id: int,
                     disable_avx512: bool = False,
                     quick_select: bool = False):
    """Register engines for ppl runtime.

    Args:
        device_id (int): -1 for cpu.
        disable_avx512 (bool): Whether to disable avx512 for x86.
        quick_select (bool): Whether to use default algorithms.
    """
    engines = []
    if device_id == -1:
        x86_options = pplnn.X86EngineOptions()
        x86_engine = pplnn.X86EngineFactory.Create(x86_options)
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


class PPLWrapper(torch.nn.Module):
    """PPLWrapper Wrapper.

    Arguments:
        model_file (str): Input onnx model file
        device_id (int): The device id to put model
    """

    def __init__(self, model_file: str, device_id: int):
        super(PPLWrapper, self).__init__()
        # enable quick select by default to speed up pipeline
        # TODO: open it to users after ppl supports saving serialized models
        # TODO: disable_avx512 will be removed or open to users in config
        engines = register_engines(
            device_id, disable_avx512=False, quick_select=True)
        runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(
            model_file, engines)
        assert runtime_builder is not None, 'Failed to create '\
            'OnnxRuntimeBuilder.'

        runtime = runtime_builder.CreateRuntime()
        assert runtime is not None, 'Failed to create the instance of Runtime.'

        self.runtime = runtime
        self.inputs = {
            runtime.GetInputTensor(i).GetName(): runtime.GetInputTensor(i)
            for i in range(runtime.GetInputCount())
        }

    def forward(self, input_data):
        """
        Arguments:
            input_data (dict): the input name and tensor pairs
        Return:
            list[np.ndarray]: list of output numpy array
        """
        for name, input_tensor in input_data.items():
            input_tensor = input_tensor.contiguous()
            self.inputs[name].ConvertFromHost(input_tensor.cpu().numpy())
        self.ppl_execute()
        outputs = []
        for i in range(self.runtime.GetOutputCount()):
            out_tensor = self.runtime.GetOutputTensor(i).ConvertToHost()
            outputs.append(np.array(out_tensor, copy=False))
        return outputs

    @TimeCounter.count_time()
    def ppl_execute(self):
        status = self.runtime.Run()
        assert status == pplcommon.RC_SUCCESS, 'Run() '\
            'failed: ' + pplcommon.GetRetCodeStr(status)
        status = self.runtime.Sync()
        assert status == pplcommon.RC_SUCCESS, 'Sync() '\
            'failed: ' + pplcommon.GetRetCodeStr(status)
