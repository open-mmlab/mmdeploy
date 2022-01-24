# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, Optional, Sequence

import onnxruntime as ort
import torch

from mmdeploy.utils import Backend, get_root_logger, parse_device_id
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper
from .init_plugins import get_ops_path


@BACKEND_WRAPPER.register_module(Backend.ONNXRUNTIME.value)
class ORTWrapper(BaseWrapper):
    """ONNXRuntime wrapper for inference.

     Args:
         onnx_file (str): Input onnx model file.
         device (str): The device to input model.
         output_names (Sequence[str] | None): Names of model outputs in order.
            Defaults to `None` and the wrapper will load the output names from
            model.

     Examples:
         >>> from mmdeploy.backend.onnxruntime import ORTWrapper
         >>> import torch
         >>>
         >>> onnx_file = 'model.onnx'
         >>> model = ORTWrapper(onnx_file, -1)
         >>> inputs = dict(input=torch.randn(1, 3, 224, 224, device='cpu'))
         >>> outputs = model(inputs)
         >>> print(outputs)
    """

    def __init__(self,
                 onnx_file: str,
                 device: str,
                 output_names: Optional[Sequence[str]] = None):
        # get the custom op path
        ort_custom_op_path = get_ops_path()
        session_options = ort.SessionOptions()
        # register custom op for onnxruntime
        logger = get_root_logger()
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
            logger.info(f'Successfully loaded onnxruntime custom ops from \
            {ort_custom_op_path}')
        else:
            logger.warning(f'The library of onnxruntime custom ops does \
            not exist: {ort_custom_op_path}')
        device_id = parse_device_id(device)
        is_cuda_available = ort.get_device() == 'GPU'
        providers = [('CUDAExecutionProvider', {'device_id': device_id})] \
            if is_cuda_available else ['CPUExecutionProvider']
        sess = ort.InferenceSession(
            onnx_file, session_options, providers=providers)
        if output_names is None:
            output_names = [_.name for _ in sess.get_outputs()]
        self.sess = sess
        self.io_binding = sess.io_binding()
        self.device_id = device_id
        self.is_cuda_available = is_cuda_available
        self.device_type = 'cuda' if is_cuda_available else 'cpu'
        super().__init__(output_names)

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Returns:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        for name, input_tensor in inputs.items():
            # set io binding for inputs/outputs
            input_tensor = input_tensor.contiguous()
            if not self.is_cuda_available:
                input_tensor = input_tensor.cpu()
            element_type = input_tensor.numpy().dtype
            self.io_binding.bind_input(
                name=name,
                device_type=self.device_type,
                device_id=self.device_id,
                element_type=element_type,
                shape=input_tensor.shape,
                buffer_ptr=input_tensor.data_ptr())

        for name in self._output_names:
            self.io_binding.bind_output(name)
        # run session to get outputs
        self.__ort_execute(self.io_binding)
        output_list = self.io_binding.copy_outputs_to_cpu()
        outputs = {}
        for output_name, numpy_tensor in zip(self._output_names, output_list):
            outputs[output_name] = torch.from_numpy(numpy_tensor)

        return outputs

    @TimeCounter.count_time()
    def __ort_execute(self, io_binding: ort.IOBinding):
        """Run inference with ONNXRuntime session.

        Args:
            io_binding (ort.IOBinding): To bind input/output to a specified
                device, e.g. GPU.
        """
        self.sess.run_with_iobinding(io_binding)
