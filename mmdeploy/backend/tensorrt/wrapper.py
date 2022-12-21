# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional, Sequence, Union

import tensorrt as trt
import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper
from .init_plugins import load_tensorrt_plugin
from .torch_allocator import TorchAllocator
from .utils import load


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    """Convert pytorch dtype to TensorRT dtype.

    Args:
        dtype (str.DataType): The data type in tensorrt.

    Returns:
        torch.dtype: The corresponding data type in torch.
    """

    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')


def torch_device_from_trt(device: trt.TensorLocation):
    """Convert pytorch device to TensorRT device.

    Args:
        device (trt.TensorLocation): The device in tensorrt.
    Returns:
        torch.device: The corresponding device in torch.
    """
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError(f'{device} is not supported by torch')


@BACKEND_WRAPPER.register_module(Backend.TENSORRT.value)
class TRTWrapper(BaseWrapper):
    """TensorRT engine wrapper for inference.

    Args:
        engine (tensorrt.ICudaEngine): TensorRT engine to wrap.
        output_names (Sequence[str] | None): Names of model outputs  in order.
            Defaults to `None` and the wrapper will load the output names from
            model.

    Note:
        If the engine is converted from onnx model. The input_names and
        output_names should be the same as onnx model.

    Examples:
        >>> from mmdeploy.backend.tensorrt import TRTWrapper
        >>> engine_file = 'resnet.engine'
        >>> model = TRTWrapper(engine_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None,
                 device_id: int = 0):
        super().__init__(output_names)
        load_tensorrt_plugin()
        self.engine = engine
        self.allocator = TorchAllocator(device_id)
        if isinstance(self.engine, str):
            self.engine = load(engine, self.allocator)

        if not isinstance(self.engine, trt.ICudaEngine):
            raise TypeError('`engine` should be str or trt.ICudaEngine,'
                            f'but given: {type(self.engine)}')

        self._register_state_dict_hook(TRTWrapper.__on_state_dict)
        self.context = self.engine.create_execution_context()

        if hasattr(self.context, 'temporary_allocator'):
            self.context.temporary_allocator = self.allocator

        self.__load_io_names()

    def __load_io_names(self):
        """Load input/output names from engine."""
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def __on_state_dict(self, state_dict: Dict[str, Any], prefix: str):
        """State dict hook
        Args:
            state_dict (Dict[str, Any]): A dict to save state information
                such as the serialized engine, input/output names.
            prefix (str): A string to be prefixed at the key of the
                state dict.
        """
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
        state_dict[prefix + 'input_names'] = self._input_names
        state_dict[prefix + 'output_names'] = self._output_names

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Return:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))

        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()

        self.__trt_execute(bindings=bindings)

        return outputs

    @TimeCounter.count_time(Backend.TENSORRT.value)
    def __trt_execute(self, bindings: Sequence[int]):
        """Run inference with TensorRT.

        Args:
            bindings (list[int]): A list of integer binding the input/output.
        """
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
