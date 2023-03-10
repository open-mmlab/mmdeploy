# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import os
import os.path as osp
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from subprocess import call, run
from typing import Any, List, Optional, Sequence

from mmdeploy.ir.onnx import ONNXParam
from mmdeploy.utils import get_root_logger
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    import_custom_modules)


@dataclass
class SNPEParam(BaseBackendParam):
    """SNPE backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        uri (str): The uri of remote device.
    """
    _default_postfix = '.dlc'

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str), ('Expect string work_dir, '
                                                f'got {self.work_dir}')
        assert isinstance(self.file_name, str), ('Expect string file_name, '
                                                 f'got {self.file_name}')
        file_name = self.file_name
        return osp.join(self.work_dir, file_name)


_BackendParam = SNPEParam


@BACKEND_MANAGERS.register('snpe', param=_BackendParam, ir_param=ONNXParam)
class SNPEManager(BaseBackendManager):

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        from .onnx2dlc import get_onnx2dlc_path
        onnx2dlc = get_onnx2dlc_path()
        if onnx2dlc is None:
            return False
        if not osp.exists(onnx2dlc):
            return False

        ret_code = call([onnx2dlc, '-v'],
                        stdout=open(os.devnull, 'wb'),
                        stderr=open(os.devnull, 'wb'))
        return ret_code == 0

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        from .onnx2dlc import get_onnx2dlc_path
        onnx2dlc = get_onnx2dlc_path()
        snpe_net_run_path = osp.join(osp.split(onnx2dlc)[0], 'snpe-net-run')
        if not osp.exists(snpe_net_run_path):
            return ''

        command = [snpe_net_run_path, '--version']
        result = run(
            command,
            stdout=open(os.devnull, 'wb'),
            stderr=open(os.devnull, 'wb'),
            universal_newlines=True)
        if result.returncode != 0:
            return ''
        else:
            return result.stdout[5:]

    @classmethod
    def to_backend(cls, onnx_path: str, save_path: str) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            onnx_path (str): The ONNX model to be converted.
            output_path (str): The output file.
        """
        from .onnx2dlc import from_onnx
        logger = get_root_logger()

        if not cls.is_available():
            logger.error('snpe support is not available, please check\n'
                         '1) `snpe-onnx-to-dlc` existed in `PATH`\n'
                         '2) snpe only support\n'
                         'ubuntu18.04')
            sys.exit(1)
        from_onnx(onnx_path, save_path)

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
        model_path = param.get_model_files()

        cls.to_backend(ir_model, model_path)

    @classmethod
    def build_wrapper(cls, model_path: str, uri: Optional[str] = None):
        """Build the wrapper for the backend model.

        Args:
            model_path (str): Backend files.
            uri (str): device uri.
        """
        from .wrapper import SNPEWrapper
        return SNPEWrapper(dlc_file=model_path, uri=uri)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        model_path = param.get_model_files()
        return cls.build_wrapper(model_path, uri=param.uri)

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

        uri = kwargs.get('uri', None)

        from .onnx2dlc import get_env_key

        if uri is not None and get_env_key() not in os.environ:
            os.environ[get_env_key()] = uri

        backend_files = [] if backend_files is None else backend_files
        if len(backend_files) > 0:
            kwargs['file_name'] = backend_files[0]
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
        export_parser.add_argument(
            '--custom-modules',
            type=str,
            nargs='*',
            help='Import custom modules.')

        parsed_args = parser.parse_args(args)
        yield parsed_args
        import_custom_modules(parsed_args.custom_modules)

        # perform command
        command = parsed_args._command

        if command == 'convert':
            # convert model
            param = _BackendParam(
                work_dir=parsed_args.work_dir,
                file_name=parsed_args.file_name,
                uri=parsed_args.uri)

            cls.to_backend_from_param(parsed_args.onnx_path, param)
