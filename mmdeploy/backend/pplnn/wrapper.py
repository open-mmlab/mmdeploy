# Copyright (c) OpenMMLab. All rights reserved.
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import onnx
import pyppl.common as pplcommon
import pyppl.nn as pplnn
import torch

from mmdeploy.utils import Backend, get_root_logger, parse_device_id
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper


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
        x86_options = pplnn.X86EngineOptions()
        x86_engine = pplnn.X86EngineFactory.Create(x86_options)
        if not x86_engine:
            logger.error('Failed to create x86 engine')
            sys.exit(-1)

        if disable_avx512:
            status = x86_engine.Configure(pplnn.X86_CONF_DISABLE_AVX512)
            if status != pplcommon.RC_SUCCESS:
                logger.error('x86 engine Configure() failed: ' +
                             pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

        engines.append(pplnn.Engine(x86_engine))

    else:
        cuda_options = pplnn.CudaEngineOptions()
        cuda_options.device_id = device_id

        cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
        if not cuda_engine:
            logger.error('Failed to create cuda engine.')
            sys.exit(-1)

        if quick_select:
            status = cuda_engine.Configure(
                pplnn.CUDA_CONF_USE_DEFAULT_ALGORITHMS)
            if status != pplcommon.RC_SUCCESS:
                logger.error('cuda engine Configure() failed: ' +
                             pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

        if input_shapes is not None:
            status = cuda_engine.Configure(pplnn.CUDA_CONF_SET_INPUT_DIMS,
                                           input_shapes)
            if status != pplcommon.RC_SUCCESS:
                logger.error(
                    'cuda engine Configure(CUDA_CONF_SET_INPUT_DIMS) failed: '
                    + pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

        if export_algo_file is not None:
            status = cuda_engine.Configure(pplnn.CUDA_CONF_EXPORT_ALGORITHMS,
                                           export_algo_file)
            if status != pplcommon.RC_SUCCESS:
                logger.error(
                    'cuda engine Configure(CUDA_CONF_EXPORT_ALGORITHMS) '
                    'failed: ' + pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

        if import_algo_file is not None:
            status = cuda_engine.Configure(pplnn.CUDA_CONF_IMPORT_ALGORITHMS,
                                           import_algo_file)
            if status != pplcommon.RC_SUCCESS:
                logger.error(
                    'cuda engine Configure(CUDA_CONF_IMPORT_ALGORITHMS) '
                    'failed: ' + pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

        engines.append(pplnn.Engine(cuda_engine))

    return engines


@BACKEND_WRAPPER.register_module(Backend.PPLNN.value)
class PPLNNWrapper(BaseWrapper):
    """PPLNN wrapper for inference.

    Args:
        onnx_file (str): Path of input ONNX model file.
        algo_file (str): Path of PPLNN algorithm file.
        device_id (int): Device id to put model.

    Examples:
        >>> from mmdeploy.backend.pplnn import PPLNNWrapper
        >>> import torch
        >>>
        >>> onnx_file = 'model.onnx'
        >>> model = PPLNNWrapper(onnx_file, 'end2end.json', 0)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 onnx_file: str,
                 algo_file: str,
                 device: str,
                 output_names: Optional[Sequence[str]] = None,
                 **kwargs):

        # enable quick select by default to speed up pipeline
        # TODO: open it to users after pplnn supports saving serialized models

        # TODO: assert device is gpu
        device_id = parse_device_id(device)

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

        if output_names is None:
            model = onnx.load(onnx_file)
            output_names = [node.name for node in model.graph.output]

        super().__init__(output_names)

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Input name and tensor pairs.

        Return:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        for name, input_tensor in inputs.items():
            input_tensor = input_tensor.contiguous()
            self.inputs[name].ConvertFromHost(input_tensor.cpu().numpy())
        self.__pplnn_execute()
        outputs = {}
        for i in range(self.runtime.GetOutputCount()):
            out_tensor = self.runtime.GetOutputTensor(i).ConvertToHost()
            name = self.output_names[i]
            if out_tensor:
                outputs[name] = np.array(out_tensor, copy=False)
            else:
                out_shape = self.runtime.GetOutputTensor(
                    i).GetShape().GetDims()
                outputs[name] = np.random.rand(*out_shape)
            outputs[name] = torch.from_numpy(outputs[name])
        return outputs

    @TimeCounter.count_time()
    def __pplnn_execute(self):
        """Run inference with PPLNN."""
        status = self.runtime.Run()
        assert status == pplcommon.RC_SUCCESS, 'Run() failed: ' + \
            pplcommon.GetRetCodeStr(status)
