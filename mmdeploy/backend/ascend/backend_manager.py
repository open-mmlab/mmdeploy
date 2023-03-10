# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import os.path as osp
import re
from argparse import Action, ArgumentParser
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from mmdeploy.ir.onnx import ONNXParam
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    import_custom_modules)
from .onnx2ascend import AtcParam


class DynamicDimsAction(Action):
    """dynamic dims argparse action."""

    def __call__(self, parser, namespace, values, option_string=None):
        """call action."""
        args = values
        if isinstance(args, str):
            args = [args]

        pattern = r'^[^\S\n]*((?P<input_name>\w+):)?(?P<val>([+|-]?\d+)(,[+|-]?\d+)*)$'  # noqa
        ret = dict()
        for arg in args:
            arg = [i.strip() for i in arg.split(';')]
            for single_arg in arg:
                if len(single_arg) == 0:
                    continue
                m = re.match(pattern, single_arg)
                if m is None:
                    raise ValueError(f'Can not parse value: {single_arg}')
                input_name = m.group('input_name')
                val = m.group('val')
                val = val.split(',')
                val = tuple(int(v) for v in val)
                if input_name in ret:
                    raise NameError(f'value of `{input_name}` '
                                    'has been assigned more than once.')
                ret[input_name] = val

        setattr(namespace, self.dest, ret)


@dataclass
class AscendParam(BaseBackendParam):
    """Ascend backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        input_shapes (ShapeType): The Default shape of the inputs.
        dynamic_batch_size (List[str]): Set dynamic batch size.
            E.g.: "batchsize1 batchsize2 batchsize3"
        dynamic_image_size (Dict[List[int]]): Set dynamic image size.
            Separate multiple nodes with semicolons (;).
            Use double quotation marks (") to enclose each argument.
            E.g.: "input0:height0,width0;input1:height1,width1"
        dynamic_dims (Dict[List[int]]): Set dynamic dims.
            Separate multiple nodes with semicolons (;).
            Use double quotation marks (") to enclose each argument.
            E.g.: "input0:dims1_n1,dims1_n2;input1:dims2_n1,dims2_n2"
        device (str): Inference device.
    """
    _default_postfix = '.om'
    dynamic_batch_size: List[str] = None
    dynamic_image_size: List[List[int]] = None
    dynamic_dims: List[List[int]] = None

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str), ('Expect string work_dir, '
                                                f'got {self.work_dir}')
        assert isinstance(self.file_name, str), ('Expect string file_name, '
                                                 f'got {self.file_name}')
        file_name = self.file_name
        if not file_name.endswith('.om'):
            file_name = file_name + '.om'
        return osp.join(self.work_dir, file_name)

    @classmethod
    def add_argument(cls, parser: ArgumentParser, name: str, dtype: Any,
                     default: Any, desc: str):
        arg_name = f'--{name.replace("_", "-")}'
        if name == 'dynamic_image_size' or name == 'dynamic_dims':
            parser.add_argument(
                arg_name, action=DynamicDimsAction, nargs='+', help=desc)
        else:
            return super().add_argument(parser, name, dtype, default, desc)


_BackendParam = AscendParam


@BACKEND_MANAGERS.register('ascend', param=_BackendParam, ir_param=ONNXParam)
class AscendManager(BaseBackendManager):

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        return importlib.util.find_spec('acl') is not None

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('acl').version
            except Exception:
                return 'None'

    @classmethod
    def to_backend(cls, onnx_model: str, output_path: str,
                   atc_param: AtcParam) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Example:
            >>> from mmdeploy.backend.ascend.onnx2ascend import AtcParam
            >>> onnx_path = 'work_dir/end2end.onnx'
            >>> output_path = 'work_dir/end2end.om
            >>> atc_param = AtcParam(input_shapes=dict(input=[1, 3, 224, 224]))
            >>> to_backend(onnx_path, output_path, atc_param)
        Args:
            onnx_path (ModelProto|str): The path of the onnx model.
            output_path (str): Path to save model.
            atc_param (AtcParam): The input args to the atc tools.
        """
        from .onnx2ascend import from_onnx
        from_onnx(onnx_model, output_path, atc_param)

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

        input_shapes = param.input_shapes

        atc_param = AtcParam(
            input_shapes=input_shapes,
            dynamic_batch_size=param.dynamic_batch_size,
            dynamic_image_size=param.dynamic_image_size,
            dynamic_dims=param.dynamic_dims)

        cls.to_backend(ir_model, model_path, atc_param=atc_param)

    @classmethod
    def build_wrapper(cls, model_path: str, device: str = 'cpu'):
        """Build the wrapper for the backend model.

        Args:
            model_path (str): The om model path.
            device (str, optional): The device info. Defaults to 'cpu'.
        """
        from .wrapper import AscendWrapper
        return AscendWrapper(model=model_path, device=device)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        model_path = param.get_model_files()
        device = param.device
        return cls.build_wrapper(model_path, device=device)

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

        model_inputs = get_model_inputs(config)[0]
        assert 'input_shapes' in model_inputs, (
            'Can not find model_inputs in config.')
        input_shapes = model_inputs['input_shapes']
        dynamic_batch_size = model_inputs.get('dynamic_batch_size', None)
        dynamic_image_size = model_inputs.get('dynamic_image_size', None)
        dynamic_dims = model_inputs.get('dynamic_dims', None)

        kwargs.setdefault('work_dir', work_dir)
        kwargs.setdefault('input_shapes', input_shapes)
        kwargs.setdefault('dynamic_batch_size', dynamic_batch_size)
        kwargs.setdefault('dynamic_image_size', dynamic_image_size)
        kwargs.setdefault('dynamic_dims', dynamic_dims)

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
                dynamic_batch_size=parsed_args.dynamic_batch_size,
                dynamic_image_size=parsed_args.dynamic_image_size,
                dynamic_dims=parsed_args.dynamic_dims,
                device=parsed_args.device)

            cls.to_backend_from_param(parsed_args.onnx_path, param)
