# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import re
from argparse import ArgumentParser
from collections import OrderedDict
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Union)

from mmdeploy.ir.onnx import ONNXParam
from mmdeploy.utils import get_root_logger
from ..base import BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam


@dataclass
class TensorRTParam(BaseBackendParam):
    """TensorRT backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        input_shapes (ShapeType): The Default shape of the inputs.
        min_shapes (ShapeType): The minimal shape of the inputs.
        max_shapes (ShapeType): The maximal shape of the inputs.
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
            if self.int8_algorithm.lower() not in ['entropy', 'minmax']:
                raise ValueError(
                    f'Unsupported int8 algorithm: {self.int8_algorithm}')


_BackendParam = TensorRTParam


@BACKEND_MANAGERS.register('tensorrt', param=_BackendParam, ir_param=ONNXParam)
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
                   int8_algorithm: str = 'entropy',
                   calib_data: Optional[Union[str, Iterable]] = None,
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
            int8_algorithm (str): algorithm used to perform the calibration.
            calib_data (Iterable|str): An iterable object to provide the input
                data. Or qual name of the object.
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

        int8_param = dict()
        if int8_mode:
            if int8_algorithm.lower() == 'entropy':
                int8_algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
            elif int8_algorithm.lower() == 'minmax':
                int8_algo = trt.CalibrationAlgoType.MINMAX_CALIBRATION
            else:
                raise ValueError(
                    f'Unsupported int8 algorithm: {int8_algorithm}')

            if isinstance(calib_data, str):
                from ..base import get_obj_by_qualname
                calib_data = get_obj_by_qualname(calib_data)

            int8_param = dict(calib_file=calib_data, algorithm=int8_algo)

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
    def to_backend_from_param(cls, ir_model: str, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            ir_model (str): The ir model path to perform the export.
            param (BaseBackendParam): Packed backend parameter.
        """
        param.check_param()

        assert isinstance(param, _BackendParam), ('Expect _BackendParam '
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

        cls.to_backend(
            ir_model,
            save_path,
            input_shapes=input_shapes,
            min_shapes=min_shapes,
            max_shapes=max_shapes,
            max_workspace_size=max_workspace_size,
            fp16_mode=fp16_mode,
            int8_mode=int8_mode,
            int8_algorithm=param.int8_algorithm,
            calib_data=param.quanti_data,
            device_id=device_id)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        assert isinstance(param, _BackendParam)
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
                                **kwargs) -> _BackendParam:
        """Build param from deploy config.

        Args:
            config (Any): The deploy config.
            work_dir (str): work directory of the parameters.
            backend_files (List[str]): The backend files of the model.

        Returns:
            BaseBackendParam: The packed backend parameter.
        """
        from mmdeploy.utils import (get_calib_config, get_common_config,
                                    get_model_inputs)
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

        if int8_mode:
            calib_config = get_calib_config(config)
            if calib_config is not None and calib_config.get(
                    'create_calib', False):
                from ..base import create_h5pydata_generator
                assert 'calib_file' in calib_config
                calib_path = osp.join(work_dir, calib_config['calib_file'])
                calib_data = create_h5pydata_generator(calib_path,
                                                       input_shapes)
                kwargs.setdefault('quanti_data', calib_data)

        ret = _BackendParam(
            work_dir=work_dir, file_name=backend_files[0], **kwargs)
        return ret

    @classmethod
    def parse_args(cls,
                   parser: ArgumentParser,
                   args: Optional[List[str]] = None):
        """Parse console arguments.

        Args:
            parser (ArgumentParser): The parser used to parse arguments.
            args (Optional[List[str]], optional): Arguments to be parsed. If
                not given, arguments from console will be parsed.
        """
        logger = get_root_logger()

        # parse args
        sub_parsers = parser.add_subparsers(
            title='command',
            description='Please select the command you want to perform.',
            dest='_command')

        # export model
        export_parser = sub_parsers.add_parser(
            name='convert', help='convert model from ONNX model.')
        export_parser.add_argument(
            '--onnx-path', required=True, help='ONNX model path.')
        _BackendParam.add_arguments(export_parser)
        export_parser.add_argument(
            '--custom-modules', type=str, nargs='*', help='ONNX model path.')

        parsed_args = parser.parse_args(args)
        yield parsed_args

        # perform command
        command = parsed_args._command

        custom_modules = parsed_args.custom_modules
        custom_modules = [] if custom_modules is None else custom_modules

        for qualname in custom_modules:
            try:
                importlib.import_module(qualname)
                logger.info(f'Import custom module: {qualname}')
            except Exception as e:
                logger.warning('Failed to import custom module: '
                               f'{qualname} with error: {e}')

        if command == 'convert':
            # convert model
            param = _BackendParam(
                work_dir=parsed_args.work_dir,
                file_name=parsed_args.file_name,
                device=parsed_args.device,
                min_shapes=parsed_args.min_shapes,
                input_shapes=parsed_args.input_shapes,
                max_shapes=parsed_args.max_shapes,
                max_workspace_size=parsed_args.max_workspace_size,
                fp16_mode=parsed_args.fp16_mode,
                int8_mode=parsed_args.int8_mode,
                int8_algorithm=parsed_args.int8_algorithm,
                quanti_data=parsed_args.quanti_data)

            cls.to_backend_from_param(parsed_args.onnx_path, param)
