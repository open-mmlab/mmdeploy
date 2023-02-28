# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from mmdeploy.ir.onnx import ONNXIRParam
from ..base import BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam


@dataclass
class TensorRTBackendParam(BaseBackendParam):
    """TensorRT backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        input_shapes (ShapeType): The Default shape of the inputs.
        min_shapes (ShapeType): The minimal shape of the inputs.
        max_shapes (ShapeType): The maximal shape of the inputs.
        input_names (List[str]): Names of the inputs.
        output_names (List[str]): Names of the outputs.
        device (str): Device used to perform inference.
        fp16_mode (bool): Enable fp16 mode.
        int8_mode (bool): Enable int8 quantization. Can be co-exist with
            fp16 mode.
        int8_algorithm (str): The quantization algorithm, choice from
            [`entropy`, `maxmin`]
        quanti_data (Union[Iterable, str]): Iterable object to provide the
            quantization data. Each iteration gives a dict of input name and
            correspond tensor.
        max_workspace_size (int): Extra workspace size required by the model.
            default to 1Gb.
    """
    _default_postfix = '.onnx'

    device: str = 'cuda'
    fp16_mode: bool = False
    int8_mode: bool = False
    int8_algorithm: str = 'entropy'
    max_workspace_size: int = 1 << 30

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str), ('Expect string work_dir, '
                                                f'got {self.work_dir}')
        assert isinstance(self.file_name, str), ('Expect string file_name, '
                                                 f'got {self.file_name}')
        return osp.join(self.work_dir, self.file_name)

    def check_param(self):
        """check param validation."""
        super().check_param()

        if self.int8_mode:
            if self.int8_algorithm not in ['entropy', 'maxmin']:
                raise ValueError(
                    f'Unsupported int8 algorithm: {self.int8_algorithm}')


@BACKEND_MANAGERS.register(
    'tensorrt', param=TensorRTBackendParam, ir_param=ONNXIRParam)
class TensorRTManager(BaseBackendManager):

    @classmethod
    def build_wrapper(
        cls,
        engine_path: str,
        output_names: Optional[Sequence[str]] = None,
    ):
        """Build the wrapper for the backend model.

        Args:
            engine_path (str): TensorRT engine file.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
        """

        from .wrapper import TRTWrapper
        return TRTWrapper(engine=engine_path, output_names=output_names)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        ret = importlib.util.find_spec('tensorrt') is not None

        if ret and with_custom_ops:
            from .init_plugins import get_ops_path
            ops_path = get_ops_path()
            custom_ops_exist = osp.exists(ops_path)
            ret = ret and custom_ops_exist

        return ret

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('tensorrt').version
            except Exception:
                return 'None'

    @classmethod
    def check_env(cls, log_callback: Callable = lambda _: _) -> str:
        """Check current environment.

        Returns:
            str: Info about the environment.
        """
        info = super().check_env(log_callback=log_callback)
        available = cls.is_available()
        ops_available = cls.is_available(with_custom_ops=True)
        ops_available = 'Available' if ops_available else 'NotAvailable'

        if available:
            ops_info = f'tensorrt custom ops:\t{ops_available}'
            log_callback(ops_info)
            info = f'{info}\n{ops_info}'
        return info

    @classmethod
    def to_backend(cls,
                   ir_path: str,
                   save_path: str,
                   input_shapes: Dict[str, Sequence],
                   min_shapes: Optional[Dict[str, Sequence]] = None,
                   max_shapes: Optional[Dict[str, Sequence]] = None,
                   max_workspace_size: int = 0,
                   fp16_mode: bool = False,
                   int8_mode: bool = False,
                   int8_param: Optional[dict] = None,
                   device_id: int = 0,
                   log_level: Any = None):
        """Convert intermediate representation to given backend.

        Args:
            ir_path (str or onnx.ModelProto): Input ir model to convert from.
            save_path (str): The path to save the output model.
            input_shapes (Dict[str, Sequence]): The input shapes of
                each input.
            min_shapes (Dict[str, Sequence]): The min shapes of each input.
            max_shapes (Dict[str, Sequence]): The max shapes of each input.
            max_workspace_size (int): To set max workspace size of TensorRT
                engine. some tactics and layers need large workspace.
            fp16_mode (bool): Specifying whether to enable fp16 mode.
                Defaults to `False`.
            int8_mode (bool): Specifying whether to enable int8 mode.
                Defaults to `False`.
            int8_param (dict): A dict of parameter  int8 mode
            device_id (int): Choice the device to create engine
            log_level (trt.Logger.Severity): The log level of TensorRT.
        """
        import tensorrt as trt

        from .utils import from_onnx
        if log_level is None:
            log_level = trt.Logger.ERROR

        # fill shapes
        if min_shapes is None:
            min_shapes = input_shapes
        if max_shapes is None:
            max_shapes = input_shapes

        merged_shapes = OrderedDict()
        for name, val in input_shapes.items():
            if name not in min_shapes:
                min_shapes[name] = val
            if name not in max_shapes:
                max_shapes[name] = val

            merged_shapes[name] = dict(
                opt_shape=val,
                min_shape=min_shapes[name],
                max_shape=max_shapes[name])

        # export model
        from_onnx(
            ir_path,
            save_path,
            input_shapes=merged_shapes,
            max_workspace_size=max_workspace_size,
            fp16_mode=fp16_mode,
            int8_mode=int8_mode,
            int8_param=int8_param,
            device_id=device_id,
            log_level=log_level)

    @classmethod
    def to_backend_from_param(cls, ir_model: str, param: TensorRTBackendParam):
        """Export to backend with packed backend parameter.

        Args:
            ir_model (str): The ir model path to perform the export.
            param (BaseBackendParam): Packed backend parameter.
        """
        param.check_param()

        assert isinstance(
            param, TensorRTBackendParam), ('Expect TensorRTBackendParam '
                                           f'get {type(param)}')
        assert isinstance(param.work_dir, str)
        assert isinstance(param.file_name, str)
        save_path = osp.join(param.work_dir, param.file_name)
        input_shapes = param.input_shapes
        min_shapes = param.min_shapes
        max_shapes = param.max_shapes
        max_workspace_size = param.max_workspace_size
        fp16_mode = param.fp16_mode
        int8_mode = param.int8_mode
        device = param.device

        m = re.match(r'^(cuda|CUDA)(:(?P<device_id>[0-9]+))?$', device)
        assert m is not None, f'Unsupported device {device}'
        device_id = m.groupdict().get('device_id', 0)

        # TODO: refactor TensorRT quantization
        int8_param = dict(
            calib_file=param.quanti_data,
            model_type='end2end',
            algorithm=param.int8_algorithm)

        cls.to_backend(
            ir_model,
            save_path,
            input_shapes=input_shapes,
            min_shapes=min_shapes,
            max_shapes=max_shapes,
            max_workspace_size=max_workspace_size,
            fp16_mode=fp16_mode,
            int8_mode=int8_mode,
            int8_param=int8_param,
            device_id=device_id)

    @classmethod
    def build_wrapper_from_param(cls, param: TensorRTBackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (TensorRTBackendParam): Packed backend parameter.
        """
        assert isinstance(param, TensorRTBackendParam)
        assert isinstance(param.work_dir, str)
        assert isinstance(param.file_name, str)
        model_path = osp.join(param.work_dir, param.file_name)
        output_names = param.output_names
        if len(output_names) == 0:
            output_names = None
        return cls.build_wrapper(model_path, output_names=output_names)

    @classmethod
    def build_param_from_config(cls,
                                config: Any,
                                work_dir: str,
                                backend_files: List[str] = None,
                                **kwargs) -> TensorRTBackendParam:
        """Build param from deploy config.

        Args:
            config (Any): The deploy config.
            work_dir (str): work directory of the parameters.
            backend_files (List[str]): The backend files of the model.

        Returns:
            BaseBackendParam: The packed backend parameter.
        """
        from mmdeploy.utils import get_common_config, get_model_inputs
        common_config = get_common_config(config)
        model_inputs = get_model_inputs(config)

        # get shapes
        assert len(model_inputs) == 1, ('Can not create param with '
                                        'len(model_inputs) > 1')
        shapes = model_inputs[0].get('input_shapes', {})
        min_shapes = OrderedDict()
        max_shapes = OrderedDict()
        input_shapes = OrderedDict()
        for name, vals in shapes.items():
            min_shapes[name] = vals.get('min_shape', [])
            input_shapes[name] = vals.get('opt_shape', [])
            max_shapes[name] = vals.get('max_shape', [])

        # others
        max_workspace_size = common_config.get('max_workspace_size', 0)
        fp16_mode = common_config.get('fp16_mode', False)
        int8_mode = common_config.get('int8_mode', False)

        kwargs.setdefault('min_shapes', min_shapes)
        kwargs.setdefault('max_shapes', max_shapes)
        kwargs.setdefault('input_shapes', input_shapes)
        kwargs.setdefault('max_workspace_size', max_workspace_size)
        kwargs.setdefault('fp16_mode', fp16_mode)
        kwargs.setdefault('int8_mode', int8_mode)

        ret = TensorRTBackendParam(
            work_dir=work_dir, file_name=backend_files[0], **kwargs)
        return ret
