import logging
import sys
from typing import Dict, Sequence

import numpy as np
import torch
from pyppl import common as pplcommon
from pyppl import nn as pplnn

from mmdeploy.utils.timer import TimeCounter


def register_engines(device_id: int,
                     disable_avx512: bool = False,
                     quick_select: bool = False,
                     input_shapes: Sequence[Sequence[int]] = None,
                     export_algo_file: str = None,
                     import_algo_file: str = None):
    """Register engines for ppl runtime.

    Args:
        device_id (int): Specifying device index. `-1` for cpu.
        disable_avx512 (bool): Whether to disable avx512 for x86.
            Defaults to `False`.
        quick_select (bool): Whether to use default algorithms.
            Defaults to `False`.
        input_shapes (Sequence[Sequence[int]]): shapes for PPL optimization.
        export_algo_file (str): File path for exporting PPL optimization file.
        import_algo_file (str): File path for loading PPL optimization file.

    Returns:
        list[pplnn.Engine]: A list of registered ppl engines.
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

        if input_shapes is not None:
            status = cuda_engine.Configure(pplnn.CUDA_CONF_SET_INPUT_DIMS,
                                           input_shapes)
            if status != pplcommon.RC_SUCCESS:
                logging.error(
                    'cuda engine Configure(CUDA_CONF_SET_INPUT_DIMS) failed: '
                    + pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

        if export_algo_file is not None:
            status = cuda_engine.Configure(pplnn.CUDA_CONF_EXPORT_ALGORITHMS,
                                           export_algo_file)
            if status != pplcommon.RC_SUCCESS:
                logging.error(
                    'cuda engine Configure(CUDA_CONF_EXPORT_ALGORITHMS) '
                    'failed: ' + pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

        if import_algo_file is not None:
            status = cuda_engine.Configure(pplnn.CUDA_CONF_IMPORT_ALGORITHMS,
                                           import_algo_file)
            if status != pplcommon.RC_SUCCESS:
                logging.error(
                    'cuda engine Configure(CUDA_CONF_IMPORT_ALGORITHMS) '
                    'failed: ' + pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

        engines.append(pplnn.Engine(cuda_engine))

    return engines


class PPLWrapper(torch.nn.Module):
    """PPL wrapper for inference.

    Args:
        onnx_file (str): Path of input ONNX model file.
        algo_file (str): Path of PPL algorithm file.
        device_id (int): Device id to put model.

    Examples:
        >>> from mmdeploy.apis.ppl import PPLWrapper
        >>> import torch
        >>>
        >>> onnx_file = 'model.onnx'
        >>> model = PPLWrapper(onnx_file, 'end2end.json', 0)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self, onnx_file: str, algo_file: str, device_id: int):
        super(PPLWrapper, self).__init__()
        # enable quick select by default to speed up pipeline
        # TODO: disable_avx512 will be removed or open to users in config
        engines = register_engines(
            device_id,
            disable_avx512=False,
            quick_select=False,
            import_algo_file=algo_file)
        runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(
            onnx_file, engines)
        assert runtime_builder is not None, 'Failed to create '\
            'OnnxRuntimeBuilder.'

        runtime = runtime_builder.CreateRuntime()
        assert runtime is not None, 'Failed to create the instance of Runtime.'

        self.runtime = runtime
        self.inputs = {
            runtime.GetInputTensor(i).GetName(): runtime.GetInputTensor(i)
            for i in range(runtime.GetInputCount())
        }

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Input name and tensor pairs.

        Return:
            list[np.ndarray]: A list of output numpy array.
        """
        for name, input_tensor in inputs.items():
            input_tensor = input_tensor.contiguous()
            self.inputs[name].ConvertFromHost(input_tensor.cpu().numpy())
        self.ppl_execute()
        outputs = []
        for i in range(self.runtime.GetOutputCount()):
            out_tensor = self.runtime.GetOutputTensor(i).ConvertToHost()
            if out_tensor:
                outputs.append(np.array(out_tensor, copy=False))
            else:
                out_shape = self.runtime.GetOutputTensor(
                    i).GetShape().GetDims()
                outputs.append(np.random.rand(*out_shape))
        return outputs

    @TimeCounter.count_time()
    def ppl_execute(self):
        """Run inference with PPL."""
        status = self.runtime.Run()
        assert status == pplcommon.RC_SUCCESS, 'Run() '\
            'failed: ' + pplcommon.GetRetCodeStr(status)
        status = self.runtime.Sync()
        assert status == pplcommon.RC_SUCCESS, 'Sync() '\
            'failed: ' + pplcommon.GetRetCodeStr(status)
