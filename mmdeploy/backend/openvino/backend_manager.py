# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import os.path as osp
import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from mmdeploy.ir.onnx import ONNXParam
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    FileNameDescriptor)


@dataclass
class OpenVINOParam(BaseBackendParam):
    """OpenVINO backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        bin_name (str): Serialized bin file. If not given, bin_name would be
            the same as file_name with postfix `.param`
        input_shapes (ShapeType): The Default shape of the inputs.
        output_names (List[str]): Names of the outputs.
        mo_options (str): Additional args to OpenVINO Model Optimizer.
    """
    file_name: FileNameDescriptor = FileNameDescriptor(
        default=None, postfix='.xml')
    bin_name: FileNameDescriptor = FileNameDescriptor(
        default=None, postfix='.bin', base_name='file_name')
    mo_options: str = ''

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str)
        assert isinstance(self.file_name, str)
        param_file_path = osp.join(self.work_dir, self.file_name)
        assert isinstance(self.bin_name, str)
        bin_file_path = osp.join(self.work_dir, self.bin_name)
        return param_file_path, bin_file_path


_BackendParam = OpenVINOParam


@BACKEND_MANAGERS.register('openvino', param=_BackendParam, ir_param=ONNXParam)
class OpenVINOManager(BaseBackendManager):

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        ret = importlib.util.find_spec('openvino') is not None

        return ret

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('openvino').version
            except Exception:
                return 'None'

    @classmethod
    def to_backend(cls,
                   onnx_path: str,
                   model_path: str,
                   input_info: Dict[str, Sequence],
                   output_names: Sequence[str],
                   bin_path: Optional[str] = None,
                   work_dir: Optional[str] = None,
                   mo_options: str = '') -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            onnx_path (str): The ONNX model files.
            model_path (str): The save model path.
            input_info (Dict[str, Sequence]): The dictionary about input name
                and corresponding shapes.
            output_names (str): The output names of the model.
            bin_path (str): The save weight path.
            work_dir (str): The work directory, backend files and logs should
                be saved in this directory.
            mo_options (str): Other args and flags that feeds to mo.
        """
        assert cls.is_available(), \
            'OpenVINO is not available, please install OpenVINO first.'
        from .onnx2openvino import from_onnx

        if work_dir is None:
            with tempfile.TemporaryDirectory() as work_dir:
                from_onnx(
                    onnx_path,
                    model_path,
                    input_info=input_info,
                    output_names=output_names,
                    bin_path=bin_path,
                    work_dir=work_dir,
                    mo_options=mo_options)
        else:
            from_onnx(
                onnx_path,
                model_path,
                input_info=input_info,
                output_names=output_names,
                bin_path=bin_path,
                work_dir=work_dir,
                mo_options=mo_options)

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
        assert isinstance(param.bin_name, str)
        bin_path = osp.join(param.work_dir, param.bin_name)

        input_info = param.input_shapes
        output_names = param.output_names
        mo_options = param.mo_options

        cls.to_backend(
            ir_model,
            model_path,
            input_info=input_info,
            output_names=output_names,
            bin_path=bin_path,
            work_dir=param.work_dir,
            mo_options=mo_options)

    @classmethod
    def build_wrapper(cls,
                      model_path: str,
                      bin_path: Optional[str] = None,
                      output_names: Optional[Sequence[str]] = None):
        """Build the wrapper for the backend model.

        Args:
            model_path (str): OpenVINO model path.
            bin_path (str): OpenVINO weight path.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
        """
        from .wrapper import OpenVINOWrapper
        return OpenVINOWrapper(
            model_path=model_path,
            bin_path=bin_path,
            output_names=output_names)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        param_path, bin_path = param.get_model_files()
        output_names = param.output_names
        if len(output_names) == 0:
            output_names = None
        return cls.build_wrapper(
            param_path, bin_path, output_names=output_names)

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
        from mmdeploy.apis.openvino import (get_input_info_from_cfg,
                                            get_mo_options_from_cfg)
        from mmdeploy.utils import get_ir_config
        ir_config = get_ir_config(config)
        output_names = ir_config.get('output_names', [])
        input_info = get_input_info_from_cfg(config)
        mo_options = get_mo_options_from_cfg(config)
        mo_options = mo_options.get_options()

        kwargs.setdefault('work_dir', work_dir)
        kwargs.setdefault('input_shapes', input_info)
        kwargs.setdefault('output_names', output_names)
        kwargs.setdefault('mo_options', mo_options)

        backend_files = [] if backend_files is None else backend_files
        if len(backend_files) > 0:
            kwargs['file_name'] = backend_files[0]
        if len(backend_files) > 1:
            kwargs['bin_name'] = backend_files[1]
        return _BackendParam(**kwargs)

    @classmethod
    @contextlib.contextmanager
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

        parsed_args = parser.parse_args(args)
        yield parsed_args

        # perform command
        command = parsed_args._command

        if command == 'convert':
            # convert model
            param = _BackendParam(
                work_dir=parsed_args.work_dir,
                file_name=parsed_args.file_name,
                bin_name=parsed_args.bin_name,
                input_shapes=parsed_args.input_shapes,
                output_names=parsed_args.output_names,
                mo_options=parsed_args.mo_options)

            cls.to_backend_from_param(parsed_args.onnx_path, param)
