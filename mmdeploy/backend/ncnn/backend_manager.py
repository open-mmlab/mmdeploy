# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from mmdeploy.ir.onnx import ONNXParam
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    dataclass_property)


@dataclass
class NCNNParam(BaseBackendParam):
    """NCNN backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        bin_name (str): Serialized bin file. If not given, bin_name would be
            the same as file_name with postfix `.param`
        use_vulkan (str): Perform inference with vulkan.
        precision (str): Precision of the model, `INT8` or `FP32`
    """
    _default_postfix = '.param'
    _default_bin_postfix = '.bin'
    _bin_name: str = None

    bin_name: str = None
    use_vulkan: bool = False
    precision: str = 'FP32'

    @dataclass_property
    def bin_name(self) -> str:
        """bin_name getter."""
        if self._bin_name is None and self.file_name is not None:
            # if bin name has not been given, use file name with postfix
            name = osp.splitext(self.file_name)[0]
            return name + self._default_bin_postfix
        return self._bin_name

    @bin_name.setter
    def bin_name(self, val) -> None:
        """bin_name setter."""
        if val is not None and osp.splitext(val)[1] == '':
            val = val + self._default_bin_postfix

        self._bin_name = val

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str)
        assert isinstance(self.file_name, str)
        param_file_path = osp.join(self.work_dir, self.file_name)
        assert isinstance(self.bin_name, str)
        bin_file_path = osp.join(self.work_dir, self.bin_name)
        return param_file_path, bin_file_path


@BACKEND_MANAGERS.register('ncnn', param=NCNNParam, ir_param=ONNXParam)
class NCNNManager(BaseBackendManager):

    @classmethod
    def build_wrapper(cls,
                      param_path: str,
                      bin_path: str,
                      output_names: Optional[Sequence[str]] = None,
                      use_vulkan: bool = False):
        """Build the wrapper for the backend model.

        Args:
            param_path (str): ncnn parameter file path.
            bin_path (str): ncnn bin file path.
            device (str, optional): The device info. Defaults to 'cpu'.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
            use_vulkan (str): Perform inference with vulkan.
        """
        from .wrapper import NCNNWrapper

        # For unittest deploy_config will not pass into _build_wrapper
        # function.
        return NCNNWrapper(
            param_file=param_path,
            bin_file=bin_path,
            output_names=output_names,
            use_vulkan=use_vulkan)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib

        from .init_plugins import get_onnx2ncnn_path, get_ops_path
        has_pyncnn = importlib.util.find_spec('ncnn') is not None
        onnx2ncnn = get_onnx2ncnn_path()
        ret = has_pyncnn and (onnx2ncnn is not None)

        if ret and with_custom_ops:
            has_pyncnn_ext = importlib.util.find_spec(
                'mmdeploy.backend.ncnn.ncnn_ext') is not None
            op_path = get_ops_path()
            custom_ops_exist = osp.exists(op_path)
            ret = ret and has_pyncnn_ext and custom_ops_exist

        return ret

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('ncnn').version
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
            ops_info = f'ncnn custom ops:\t{ops_available}'
            log_callback(ops_info)
            info = f'{info}\n{ops_info}'

        return info

    @classmethod
    def to_backend(cls, onnx_path: str, param_path: str,
                   bin_path: str) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            onnx_path (str): The ONNX model path.
            param_path (str): ncnn parameter file path.
            bin_path (str): ncnn bin file path.
            work_dir (str): The work directory, backend files and logs should
                be saved in this directory.
        Returns:
            Sequence[str]: Backend files.
        """
        from mmdeploy.utils import get_root_logger
        logger = get_root_logger()

        if not cls.is_available():
            logger.error('ncnn support is not available, please make sure:\n'
                         '1) `mmdeploy_onnx2ncnn` existed in `PATH`\n'
                         '2) python import ncnn success')
            sys.exit(1)

        from .onnx2ncnn import from_onnx

        from_onnx(onnx_path, param_path, bin_path)

    @classmethod
    def to_backend_from_param(cls, ir_model: str, param: NCNNParam):
        """Export to backend with packed backend parameter.

        Args:
            ir_model (str): The ir model path to perform the export.
            param (BaseBackendParam): Packed backend parameter.
        """
        assert isinstance(param, NCNNParam)
        assert isinstance(param.work_dir, str)
        assert isinstance(param.file_name, str)
        model_path = osp.join(param.work_dir, param.file_name)
        assert isinstance(param.bin_name, str)
        bin_path = osp.join(param.work_dir, param.bin_name)

        cls.to_backend(ir_model, model_path, bin_path)

    @classmethod
    def build_wrapper_from_param(cls, param: NCNNParam):
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
                                **kwargs) -> NCNNParam:
        """Build param from deploy config.

        Args:
            config (Any): The deploy config.
            work_dir (str): work directory of the parameters.
            backend_files (List[str]): The backend files of the model.

        Returns:
            BaseBackendParam: The packed backend parameter.
        """
        from mmdeploy.utils import get_backend_config
        backend_cfg = get_backend_config(config)
        use_vulkan = backend_cfg.get('use_vulkan', False)
        kwargs.update(dict(work_dir=work_dir, use_vulkan=use_vulkan))

        backend_files = [] if backend_files is None else backend_files
        if len(backend_files) > 0:
            kwargs['file_name'] = backend_files[0]
        if len(backend_files) > 1:
            kwargs['bin_name'] = backend_files[1]
        return NCNNParam(**kwargs)

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
            title='action',
            description='Please select the action you want to perform.',
            dest='_action')

        # export model
        export_parser = sub_parsers.add_parser(
            name='convert', help='convert ncnn model from ONNX model.')
        export_parser.add_argument(
            '--onnx-path', required=True, help='ONNX model path.')
        NCNNParam.add_arguments(export_parser)

        parsed_args = parser.parse_args(args)
        yield parsed_args

        # perform action
        action = parsed_args._action

        if action == 'convert':
            # convert model
            param = NCNNParam(
                work_dir=parsed_args.work_dir,
                file_name=parsed_args.file_name,
                bin_name=parsed_args.bin_name,
                precision=parsed_args.precision,
                use_vulkan=parsed_args.use_vulkan)

            cls.to_backend_from_param(parsed_args.onnx_path, param)
