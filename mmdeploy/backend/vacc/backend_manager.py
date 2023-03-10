# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import os.path as osp
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from mmdeploy.ir.onnx import ONNXParam
from mmdeploy.utils import get_root_logger
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    get_obj_by_qualname, import_custom_modules)


@dataclass
class VACCParam(BaseBackendParam):
    """VACC backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        quant_mode (str): quantization mode, choice between ['fp16', 'int8']
        input_shapes (ShapeType): The Default shape of the inputs.
        output_names (List[str]): Names of the outputs.
        calib_num (int): Max numbers of calibration data.
        qconfig (Dict): Dictionary arguments feed to vacc.qconfig.
            Or qualname to the dict.
        quanti_data (Any): Calibration dataset. Iterable object of
            `Dict[str, ndarray]`
        data_transmode (int): `tvm.build_config` arguments.
        cluster_mode (int): `tvm.build_config` arguments.
        vdsp_params_info (str|Dict): vdsp parameters file or qualname of the
            parameters dictionary.
    """
    quant_mode: str = 'fp16'
    calib_num: int = 1000
    qconfig: Union[str, Dict] = field(default_factory=dict)
    data_transmode: int = 1
    cluster_mode: int = 0
    vdsp_params_info: Union[str, Dict] = None

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str)
        assert isinstance(self.file_name, str)
        save_dir = '-'.join([self.file_name, self.quant_mode])
        name = osp.split(self.file_name)[1]
        model_prefix = osp.join(self.work_dir, save_dir, name)
        return [
            model_prefix + '.so', model_prefix + '.json',
            model_prefix + '.params'
        ]


_BackendParam = VACCParam


@BACKEND_MANAGERS.register('vacc', param=VACCParam, ir_param=ONNXParam)
class VACCManager(BaseBackendManager):

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib

        has_vacc = importlib.util.find_spec('vacc') is not None
        has_tvm = importlib.util.find_spec('tvm') is not None
        ret = has_vacc and has_tvm

        return ret

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('vacc').version
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
            ops_info = f'vacc custom ops:\t{ops_available}'
            log_callback(ops_info)
            info = f'{info}\n{ops_info}'

        return info

    @classmethod
    def to_backend(
        cls,
        onnx_model: str,
        output_path: str,
        model_name: str,
        input_shapes: Dict[str, Sequence],
        quant_mode: str = 'fp16',
        calib_num: int = 1000,
        qconfig: Optional[Dict] = None,
        data_transmode: int = 1,
        cluster_mode: int = 0,
    ) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            onnx_model (str): Input onnx model.
            output_path (str): File path to save VACC model.
            model_name (str): model name.
            input_shapes (ShapeType): The Default shape of the inputs.
            qconfig (Dict): Dictionary arguments feed to vacc.qconfig.
        """
        logger = get_root_logger()

        if not cls.is_available():
            logger.error(
                'vacc and tvm support is not available, please make sure:\n'
                '1) `vacc/python` and `tvm/python` existed in `PYTHONPATH`\n'
                '2) python import tvm and import vacc success')
            sys.exit(1)

        from .onnx2vacc import from_onnx

        if isinstance(qconfig, str):
            qconfig = get_obj_by_qualname(qconfig)
            assert isinstance(qconfig, Dict)

        from_onnx(
            onnx_model,
            output_path=output_path,
            model_name=model_name,
            input_shapes=input_shapes,
            quant_mode=quant_mode,
            calib_num=calib_num,
            qconfig=qconfig,
            data_transmode=data_transmode,
            cluster_mode=cluster_mode)

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

        cls.to_backend(
            ir_model,
            output_path=param.work_dir,
            model_name=param.file_name,
            input_shapes=param.input_shapes,
            quant_mode=param.quant_mode,
            calib_num=param.calib_num,
            qconfig=param.qconfig,
            data_transmode=param.data_transmode,
            cluster_mode=param.cluster_mode)

    @classmethod
    def build_wrapper(cls,
                      lib_file: str,
                      graph_file: str,
                      param_file: str,
                      vdsp_params_info: dict,
                      output_names: Optional[Sequence[str]] = None):
        """Build the wrapper for the backend model.

        Args:
            lib_file (str): Path of a model lib file.
            graph_file (str): Path of a model graph file.
            param_file (str): Path of a model param file.
            vdsp_params_info_json (str): Path of a vdsp params info json file.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
        """
        from .wrapper import VACCWrapper

        # For unittest deploy_config will not pass into _build_wrapper
        # function.

        if isinstance(vdsp_params_info,
                      str) and not osp.exists(vdsp_params_info):
            vdsp_params_info = get_obj_by_qualname(vdsp_params_info)

        try:
            return VACCWrapper(
                lib_file=lib_file,
                graph_file=graph_file,
                param_file=param_file,
                vdsp_params_info=vdsp_params_info,
                output_names=output_names)
        except Exception as e:
            print(f'failed with error: {e}')
            print('Build model process success, wrapper process stopped')
            exit(1)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        model_paths = param.get_model_files()
        output_names = param.output_names
        vdsp_params_info = param.vdsp_params_info
        return cls.build_wrapper(
            *model_paths,
            vdsp_params_info=vdsp_params_info,
            output_names=output_names)

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

        from mmdeploy.utils import (get_common_config, get_ir_config,
                                    get_model_inputs)

        deploy_cfg = config
        ir_cfg = get_ir_config(deploy_cfg)
        model_inputs = get_model_inputs(deploy_cfg)[0]
        common_params = get_common_config(deploy_cfg)
        model_name = common_params['name']
        output_names = ir_cfg.get('output_names', None)
        vdsp_params_info = common_params['vdsp_params_info']
        input_shapes = model_inputs.get('shape', None)
        qconfig = model_inputs.get('qconfig', {})
        quant_mode = qconfig.pop('dtype', 'fp16')
        calib_num = qconfig.pop('calib_num', 1000)
        data_transmode = qconfig.pop('data_transmode', 1)
        cluster_mode = qconfig.pop('cluster_mode', 1)

        kwargs.setdefault('output_names', output_names)
        kwargs.setdefault('vdsp_params_info', vdsp_params_info)
        kwargs.setdefault('input_shapes', input_shapes)
        kwargs.setdefault('qconfig', qconfig)
        kwargs.setdefault('quant_mode', quant_mode)
        kwargs.setdefault('calib_num', calib_num)
        kwargs.setdefault('data_transmode', data_transmode)
        kwargs.setdefault('cluster_mode', cluster_mode)

        kwargs.setdefault('work_dir', work_dir)
        if len(backend_files) == 1:
            file_name = model_name
        else:
            lib_path = osp.join(work_dir, backend_files[0])
            lib_dir = osp.split(lib_path)[0]
            file_name = lib_dir[:-len(quant_mode) - 1]

        kwargs.setdefault('file_name', file_name)

        return cls.build_param(**kwargs)

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
            help='Custom module path.')

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
                input_shapes=parsed_args.input_shapes,
                output_names=parsed_args.output_names,
                quant_mode=parsed_args.quant_mode,
                calib_num=parsed_args.calib_num,
                qconfig=parsed_args.qconfig,
                data_transmode=parsed_args.data_transmode,
                vdsp_params_info=parsed_args.vdsp_params_info,
                cluster_mode=parsed_args.cluster_mode)

            cls.to_backend_from_param(parsed_args.onnx_path, param)
