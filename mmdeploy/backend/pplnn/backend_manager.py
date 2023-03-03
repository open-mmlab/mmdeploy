# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from mmdeploy.ir.onnx import ONNXParam
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    dataclass_property)


@dataclass
class PPLNNParam(BaseBackendParam):
    """PPLNN backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        algo_name (str): Serialized algorithm file. If not given,
            algo_name would be he same as file_name with postfix `.json`
        input_shapes (ShapeType): The Default shape of the inputs.
        device (str): Inference device.
        disable_avx512 (bool): Whether to disable avx512 for x86.
            Defaults to `False`.
        quick_select (bool): Whether to use default algorithms.
            Defaults to `False`.
    """
    _default_postfix = '.onnx'
    _algorithm_postfix = '.json'
    _algo_name = None

    algo_name: str = None
    disable_avx512: bool = False
    quick_select: bool = False

    @dataclass_property
    def algo_name(self) -> str:
        """algo_name getter."""
        if self._algo_name is None and self.file_name is not None:
            # if bin name has not been given, use file name with postfix
            name = osp.splitext(self.file_name)[0]
            return name + self._algorithm_postfix
        return self._algo_name

    @algo_name.setter
    def algo_name(self, val) -> None:
        """algo_name setter."""
        if val is not None and osp.splitext(val)[1] == '':
            val = val + self._algorithm_postfix

        self._algo_name = val

    def get_model_files(self) -> List[str]:
        """get the model files."""
        assert isinstance(self.work_dir, str)
        assert isinstance(self.file_name, str)
        param_file_path = osp.join(self.work_dir, self.file_name)
        assert isinstance(self.algo_name, str)
        algorithm_file_path = osp.join(self.work_dir, self.algo_name)
        return param_file_path, algorithm_file_path


_BackendParam = PPLNNParam


@BACKEND_MANAGERS.register('pplnn', param=_BackendParam, ir_param=ONNXParam)
class PPLNNManager(BaseBackendManager):

    @classmethod
    def build_wrapper(cls,
                      onnx_file: str,
                      algo_file: Optional[str] = None,
                      device: str = 'cpu',
                      output_names: Optional[Sequence[str]] = None):
        """Build the wrapper for the backend model.

        Args:
            onnx_file (str): Path of input ONNX model file.
            algo_file (str): Path of PPLNN algorithm file.
            device (str, optional): The device info. Defaults to 'cpu'.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
        """
        from .wrapper import PPLNNWrapper
        return PPLNNWrapper(
            onnx_file=onnx_file,
            algo_file=algo_file if osp.exists(algo_file) else None,
            device=device,
            output_names=output_names)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        ret = importlib.util.find_spec('pyppl') is not None

        return ret

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('pyppl').version
            except Exception:
                return 'None'

    @classmethod
    def to_backend(cls,
                   onnx_file: str,
                   output_file: str,
                   algo_file: Optional[str] = None,
                   input_shapes: Optional[Dict[str, Sequence]] = None,
                   device: str = 'cpu',
                   disable_avx512: bool = False,
                   quick_select: bool = False) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            onnx_file (str): Path of input ONNX model file.
            output_file (str): Path of output ONNX model file.
            algo_file (str): Path of PPLNN algorithm file.
            input_shapes (Dict[str, Sequence[int]] | None): Shapes for PPLNN
                optimization, default to None.
            device (str, optional): The device type. Defaults to 'cpu'.
            disable_avx512 (bool): Whether to disable avx512 for x86.
                Defaults to `False`.
            quick_select (bool): Whether to use default algorithms.
                Defaults to `False`.
        Returns:
            Sequence[str]: Backend files.
        """
        from .onnx2pplnn import from_onnx
        assert cls.is_available(), \
            'PPLNN is not available, please install PPLNN first.'

        from_onnx(
            onnx_file,
            output_file,
            algo_file,
            input_shapes=input_shapes,
            device=device,
            disable_avx512=disable_avx512,
            quick_select=quick_select)

    @classmethod
    def to_backend_from_param(cls, ir_model: str, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            ir_model (str): The ir model path to perform the export.
            param (BaseBackendParam): Packed backend parameter.
        """
        assert isinstance(param, _BackendParam)
        assert isinstance(param.work_dir, str)
        assert isinstance(param.file_name, str)
        model_path = osp.join(param.work_dir, param.file_name)
        assert isinstance(param.algo_name, str)
        algo_path = osp.join(param.work_dir, param.algo_name)

        input_shapes = param.input_shapes
        device = param.device

        cls.to_backend(
            ir_model,
            model_path,
            algo_file=algo_path,
            input_shapes=input_shapes,
            device=device,
            disable_avx512=param.disable_avx512,
            quick_select=param.quick_select)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        model_path, algo_path = param.get_model_files()
        output_names = param.output_names
        if len(output_names) == 0:
            output_names = None
        device = param.device
        return cls.build_wrapper(
            model_path, algo_path, device=device, output_names=output_names)

    @classmethod
    def build_param_from_config(cls,
                                config: Any,
                                work_dir: str,
                                backend_files: Sequence[str] = None,
                                **kwargs) -> _BackendParam:
        """Build param from deploy config.

        Args:
            config (Any): The deploy config.
            work_dir (str): work directory of the parameters.
            backend_files (List[str]): The backend files of the model.

        Returns:
            BaseBackendParam: The packed backend parameter.
        """
        from mmdeploy.utils import get_model_inputs
        model_inputs = get_model_inputs(config)
        input_shapes = model_inputs.get('opt_shape', [1, 3, 224, 224])
        input_shapes = [input_shapes]

        kwargs.setdefault('work_dir', work_dir)
        kwargs.setdefault('input_shapes', input_shapes)

        backend_files = [] if backend_files is None else backend_files
        if len(backend_files) > 0:
            kwargs['file_name'] = backend_files[0]
        if len(backend_files) > 1:
            kwargs['algo_name'] = backend_files[1]
        return _BackendParam(**kwargs)

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

        if command == 'convert':
            # convert model
            param = _BackendParam(
                work_dir=parsed_args.work_dir,
                file_name=parsed_args.file_name,
                algo_name=parsed_args.algo_name,
                input_shapes=parsed_args.input_shapes,
                device=parsed_args.device,
                disable_avx512=parsed_args.disable_avx512,
                quick_select=parsed_args.quick_select)

            cls.to_backend_from_param(parsed_args.onnx_path, param)
