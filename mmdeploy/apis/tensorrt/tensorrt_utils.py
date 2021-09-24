from typing import Dict, Sequence, Union

import onnx
import tensorrt as trt
import torch
from packaging import version

from mmdeploy.utils.timer import TimeCounter
from .calib_utils import HDF5Calibrator


def create_trt_engine(onnx_model: Union[str, onnx.ModelProto],
                      input_shapes: Dict[str, Sequence[int]],
                      log_level: trt.Logger.Severity = trt.Logger.ERROR,
                      fp16_mode: bool = False,
                      int8_mode: bool = False,
                      int8_param: dict = None,
                      max_workspace_size: int = 0,
                      device_id: int = 0,
                      **kwargs):
    """Create a tensorrt engine from ONNX.

    Args:
        onnx_model (str or onnx.ModelProto): Input onnx model to convert from.
        input_shapes (Dict[str, Sequence[int]]): The min/opt/max shape of
            each input.
        log_level (trt.Logger.Severity): The log level of TensorRT. Defaults to
            `trt.Logger.ERROR`.
        fp16_mode (bool): Specifying whether to enable fp16 mode.
            Defaults to `False`.
        int8_mode (bool): Specifying whether to enable int8 mode.
            Defaults to `False`.
        int8_param (dict): A dict of parameter  int8 mode. Defaults to `None`.
        max_workspace_size (int): To set max workspace size of TensorRT engine.
            some tactics and layers need large workspace. Defaults to `0`.
        device_id (int): Choice the device to create engine. Defaults to `0`.

    Returns:
        tensorrt.ICudaEngine: The TensorRT engine created from onnx_model.

    Example:
        >>> from mmdeploy.apis.tensorrt import create_trt_engine
        >>> engine = create_trt_engine(
        >>>             "onnx_model.onnx",
        >>>             {'input': {"min_shape" : [1, 3, 160, 160],
        >>>                        "opt_shape" : [1, 3, 320, 320],
        >>>                        "max_shape" : [1, 3, 640, 640]}},
        >>>             log_level=trt.Logger.WARNING,
        >>>             fp16_mode=True,
        >>>             max_workspace_size=1 << 30,
        >>>             device_id=0)
        >>>             })
    """
    device = torch.device('cuda:{}'.format(device_id))
    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx: {onnx_model}\n{error_msgs}')

    # config builder
    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode:
        config.set_flag(trt.BuilderFlag.INT8)
        assert int8_param is not None
        config.int8_calibrator = HDF5Calibrator(
            int8_param['calib_file'],
            input_shapes,
            model_type=int8_param['model_type'],
            device_id=device_id,
            algorithm=int8_param.get(
                'algorithm', trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2))

        builder.int8_mode = int8_mode
        builder.int8_calibrator = config.int8_calibrator

    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'
    return engine


def save_trt_engine(engine: trt.ICudaEngine, path: str):
    """Serialize TensorRT engine to disk.

    Args:
        engine (tensorrt.ICudaEngine): TensorRT engine to be serialized.
        path (str): The disk path to write the engine.
    """
    with open(path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))


def load_trt_engine(path: str):
    """Deserialize TensorRT engine from disk.

    Args:
        path (str): The disk path to read the engine.

    Returns:
        tensorrt.ICudaEngine: The TensorRT engine loaded from disk.
    """
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine


def torch_dtype_from_trt(dtype: trt.DataType):
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


class TRTWrapper(torch.nn.Module):
    """TensorRT engine wrapper for inference.

    Args:
        engine (tensorrt.ICudaEngine): TensorRT engine to wrap.

    Note:
        If the engine is converted from onnx model. The input_names and
        output_names should be the same as onnx model.

    Examples:
        >>> from mmdeploy.apis.tensorrt import TRTWrapper
        >>> engine_file = 'resnet.engine'
        >>> model = TRTWrapper(engine_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self, engine: Union[str, trt.ICudaEngine]):
        super(TRTWrapper, self).__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            self.engine = load_trt_engine(engine)

        if not isinstance(self.engine, trt.ICudaEngine):
            raise TypeError(f'`engine` should be str or trt.ICudaEngine, \
                but given: {type(self.engine)}')

        self._register_state_dict_hook(TRTWrapper._on_state_dict)
        self.context = self.engine.create_execution_context()

        self._load_io_names()

    def _load_io_names(self):
        # get input and output names from engine
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        output_names = list(set(names) - set(input_names))
        self.input_names = input_names
        self.output_names = output_names

    def _on_state_dict(self, state_dict, prefix):
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
        state_dict[prefix + 'input_names'] = self.input_names
        state_dict[prefix + 'output_names'] = self.output_names

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        engine_bytes = state_dict[prefix + 'engine']

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()

        self.input_names = state_dict[prefix + 'input_names']
        self.output_names = state_dict[prefix + 'output_names']

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Return:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        assert self.input_names is not None
        assert self.output_names is not None
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for input_name, input_tensor in inputs.items():
            idx = self.engine.get_binding_index(input_name)

            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()

        self.trt_execute(bindings=bindings)

        return outputs

    @TimeCounter.count_time()
    def trt_execute(self, bindings: Sequence[int]):
        """Run inference with TensorRT.

        Args:
            bindings (list[int]): A list of integer binding the input/output.
        """
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
