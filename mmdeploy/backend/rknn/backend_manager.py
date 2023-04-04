# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import os.path as osp
import re
from argparse import Action, ArgumentParser
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from mmdeploy.ir.onnx import ONNXParam
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    FileNameDescriptor, import_custom_modules)


class MeanStdAction(Action):
    """dtype argparse action."""

    def __call__(self, parser, namespace, values, option_string=None):
        """call action."""
        args = values
        if isinstance(args, str):
            args = [args]

        pattern = r'^[^\S\n]*((?P<input_name>\w+):)?(?P<val>(.+)(,.+)*)$'
        ret = dict()
        for arg in args:
            if len(arg) == 0:
                continue
            m = re.match(pattern, arg)
            if m is None:
                raise ValueError(f'Can not parse value: {arg}')
            input_name = m.group('input_name')
            val = m.group('val')
            val = val.split(',')
            val = tuple(float(v) for v in val)
            if input_name in ret:
                raise NameError(f'value of `{input_name}` has been assigned'
                                'more than once.')
            ret[input_name] = val

        setattr(namespace, self.dest, ret)


@dataclass
class RKNNParam(BaseBackendParam):
    """RKNN backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        input_shapes (ShapeType): The Default shape of the inputs.
        input_names (List[str]): Names of the inputs.
        output_names (List[str]): Names of the outputs.
        mean_values (Dict[str, List[int]]): mean of the inputs. with format:
            `input_name1:v0,v1,v2 input_name2:v0,v1,v2`
        std_values (Dict[str, List[int]]): mean of the inputs. with format:
            `input_name1:v0,v1,v2 input_name2:v0,v1,v2`
        device (str): Target platform, such as `rv1126` or `rk3588`.
        optimization_level (int): The optimization level of model. Default to 1
        do_quantization (bool): Enable model quantization.
        dataset (str): Dataset file. Each line is an image path.
        pre_compile (bool): Pre compile the model (smaller size and load
            quicker, but can't run on simulator)
    """

    file_name: FileNameDescriptor = FileNameDescriptor(
        default=None, postfix='.rknn')
    mean_values: Dict[str, List[int]] = None
    std_values: Dict[str, List[int]] = None
    optimization_level: int = 1
    do_quantization: bool = False
    dataset: str = None
    pre_compile: bool = False

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str), ('Expect string work_dir, '
                                                f'got {self.work_dir}')
        assert isinstance(self.file_name, str), ('Expect string file_name, '
                                                 f'got {self.file_name}')
        return osp.join(self.work_dir, self.file_name)

    @classmethod
    def add_argument(cls, parser: ArgumentParser, name: str, dtype: Any,
                     default: Any, desc: str):
        arg_name = f'--{name.replace("_", "-")}'
        if name == 'mean_values' or name == 'std_values':
            parser.add_argument(
                arg_name, action=MeanStdAction, nargs='+', help=desc)
        else:
            return super().add_argument(parser, name, dtype, default, desc)


_BackendParam = RKNNParam


@BACKEND_MANAGERS.register('rknn', param=_BackendParam, ir_param=ONNXParam)
class RKNNManager(BaseBackendManager):

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        try:
            ret = importlib.util.find_spec('rknn') is not None
        except Exception:
            pass
        return ret

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            rknn_version = None
            rknn2_version = None
            try:
                rknn_version = pkg_resources.get_distribution(
                    'rknn-toolkit').version
            except Exception:
                pass
            try:
                rknn2_version = pkg_resources.get_distribution(
                    'rknn-toolkit2').version
            except Exception:
                pass
            if rknn2_version is not None:
                return rknn2_version
            elif rknn_version is not None:
                return rknn_version
            return 'None'

    @classmethod
    def check_env(cls, log_callback: Callable = lambda _: _) -> str:
        """Check current environment.

        Returns:
            str: Info about the environment.
        """
        import pkg_resources
        try:
            rknn_version = 'None'
            rknn2_version = 'None'
            try:
                rknn_version = pkg_resources.get_distribution(
                    'rknn-toolkit').version
            except Exception:
                pass
            try:
                rknn2_version = pkg_resources.get_distribution(
                    'rknn-toolkit2').version
            except Exception:
                pass

            rknn_info = f'rknn-toolkit:\t{rknn_version}'
            rknn2_info = f'rknn-toolkit2:\t{rknn2_version}'
            log_callback(rknn_info)
            log_callback(rknn2_info)

            info = '\n'.join([rknn_info, rknn2_info])

        except Exception:
            info = f'{cls.backend_name}:\tCheckFailed'
            log_callback(info)
        return info

    @classmethod
    def to_backend(cls,
                   onnx_path: str,
                   output_path: str,
                   input_names: List[str],
                   output_names: List[str],
                   input_shapes: Dict[str, Sequence],
                   rknn_config: Any,
                   do_quantization: bool = False,
                   dataset: Optional[str] = None,
                   pre_compile: bool = False) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            onnx_path (str): The intermediate representation files.
            output_path (str): File path to save RKNN model.
            input_names (List[str]): Names of the inputs.
            output_names (List[str]): Names of the outputs.
            input_shapes (ShapeType): The Default shape of the inputs.
            rknn_config (RKNNConfig): Config of the rknn toolset. Defined in
                `mmdeploy.backend.rknn.onnx2rknn`.
            optimization_level (int): The optimization level of model.
                Default to 1
            do_quantization (bool): Enable model quantization.
            dataset (str): Dataset file. Each line is an image path.
            pre_compile (bool): Pre compile the model (smaller size and load
                quicker, but can't run on simulator)
        Returns:
            Sequence[str]: Backend files.
        """
        assert cls.is_available(
        ), 'RKNN is not available, please install RKNN first.'

        from .onnx2rknn import onnx2rknn

        onnx2rknn(
            onnx_path,
            output_path,
            input_names,
            output_names,
            input_shapes,
            rknn_config=rknn_config,
            do_quantization=do_quantization,
            dataset=dataset,
            pre_compile=pre_compile)

    @classmethod
    def to_backend_from_param(cls, ir_model: str, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            ir_model (str): The ir model path to perform the export.
            param (BaseBackendParam): Packed backend parameter.
        """
        from .onnx2rknn import RKNNConfig

        assert isinstance(param, _BackendParam)
        assert isinstance(param.work_dir, str)
        assert isinstance(param.file_name, str)
        model_path = osp.join(param.work_dir, param.file_name)

        input_shapes = param.input_shapes
        device = param.device

        # get input names
        input_names = param.input_names
        output_names = param.output_names
        mean_values = param.mean_values
        if mean_values is not None:
            mean_values = list(param.mean_values[name] for name in input_names
                               if name in param.mean_values)
        std_values = param.std_values
        if std_values is not None:
            std_values = list(param.std_values[name] for name in input_names
                              if name in param.std_values)
        optimization_level = param.optimization_level
        target_platform = device
        rknn_config = RKNNConfig(
            mean_values=mean_values,
            std_values=std_values,
            optimization_level=optimization_level,
            target_platform=target_platform)

        do_quantization = param.do_quantization
        dataset = param.dataset
        pre_compile = param.pre_compile

        cls.to_backend(
            ir_model,
            model_path,
            input_names,
            output_names,
            input_shapes=input_shapes,
            rknn_config=rknn_config,
            do_quantization=do_quantization,
            dataset=dataset,
            pre_compile=pre_compile)

    @classmethod
    def build_wrapper(cls,
                      model_path: str,
                      target_platform: str,
                      input_names: Optional[Sequence[str]] = None,
                      output_names: Optional[Sequence[str]] = None):
        """Build the wrapper for the backend model.

        Args:
            model_path (str): Backend model file.
            target_platform (str): Target platform, such as `rv1126` or
                `rk3588`.
            input_names (Optional[Sequence[str]], optional): input names.
                Defaults to None.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
        """
        from .wrapper import RKNNWrapper
        return RKNNWrapper(
            model=model_path,
            target_platform=target_platform,
            input_names=input_names,
            output_names=output_names)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        model_path = param.get_model_files()
        input_names = param.input_names
        output_names = param.output_names
        device = param.device
        return cls.build_wrapper(
            model_path,
            target_platform=device,
            input_names=input_names,
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
                                    get_quantization_config)
        from mmdeploy.utils.config_utils import get_backend_config

        deploy_cfg = config
        common_params = get_common_config(deploy_cfg)
        onnx_params = get_ir_config(deploy_cfg)
        quantization_cfg = get_quantization_config(deploy_cfg)

        input_names = onnx_params.get('input_names', None)
        output_names = onnx_params.get('output_names', None)
        input_size_list = get_backend_config(deploy_cfg).get(
            'input_size_list', None)

        mean_values = common_params.get('mean_values', None)
        std_values = common_params.get('std_values', None)
        target_platform = common_params['target_platform']
        optimization_level = common_params['optimization_level']
        if mean_values is not None:
            mean_values = dict(zip(input_names, mean_values))
        if std_values is not None:
            std_values = dict(zip(input_names, std_values))

        do_quantization = quantization_cfg.get('do_quantization', False)
        dataset = quantization_cfg.get('dataset', None)
        pre_compile = quantization_cfg.get('pre_compile', False)
        rknn_batch_size = quantization_cfg.get('rknn_batch_size', -1)

        batched_input_size_list = list(
            (rknn_batch_size, *size) for size in input_size_list)
        input_shapes = dict(zip(input_names, batched_input_size_list))

        kwargs.setdefault('input_names', input_names)
        kwargs.setdefault('output_names', output_names)
        kwargs.setdefault('input_shapes', input_shapes)

        kwargs.setdefault('mean_values', mean_values)
        kwargs.setdefault('std_values', std_values)
        kwargs['device'] = target_platform
        kwargs.setdefault('optimization_level', optimization_level)

        kwargs.setdefault('do_quantization', do_quantization)
        kwargs.setdefault('dataset', dataset)
        kwargs.setdefault('pre_compile', pre_compile)

        kwargs.setdefault('work_dir', work_dir)
        kwargs.setdefault('input_shapes', input_shapes)

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
                input_shapes=parsed_args.input_shapes,
                input_names=parsed_args.input_names,
                output_names=parsed_args.output_names,
                mean_values=parsed_args.mean_values,
                std_values=parsed_args.std_values,
                device=parsed_args.device,
                optimization_level=parsed_args.optimization_level,
                do_quantization=parsed_args.do_quantization,
                dataset=parsed_args.dataset,
                pre_compile=parsed_args.pre_compile)

            cls.to_backend_from_param(parsed_args.onnx_path, param)
