# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import os.path as osp
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from mmdeploy.ir.torchscript import TorchScriptParam
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    FileNameDescriptor, import_custom_modules)


@dataclass
class CoreMLParam(BaseBackendParam):
    """CoreML backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        input_shapes (ShapeType): The Default shape of the inputs.
        min_shapes (ShapeType): The minimal shape of the inputs.
        max_shapes (ShapeType): The maximal shape of the inputs.
        input_names (List[str]): Names of the inputs.
        output_names (List[str]): Names of the outputs.
        compute_precision (str): The model precision, FLOAT16 or FLOAT32,
            read coremltools.precision for more detail, default `FLOAT32`.
        convert_to (str): The converted model type, can be
            'neuralnetwork' or 'mlprogram'. Defaults to 'neuralnetwork'.
        minimum_deployment_target (str, optional): minimum deploy target.
            iOS15, iOS16, etc., see coremltools.target
        skip_model_load (bool, optional): Skip model load.
            Defaults to True.
    """
    file_name: FileNameDescriptor = FileNameDescriptor(
        default=None, postfix='.mlpackage')
    compute_precision: str = 'FLOAT32'
    convert_to: str = 'mlprogram'
    minimum_deployment_target: Optional[str] = None
    skip_model_load: bool = True

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str), ('Expect string work_dir, '
                                                f'got {self.work_dir}')
        assert isinstance(self.file_name, str), ('Expect string file_name, '
                                                 f'got {self.file_name}')
        file_name = self.file_name
        return osp.join(self.work_dir, file_name)

    def check_param(self):
        """Check the parameter validation."""
        if self.convert_to == 'mlprogram' and not self.file_name.endswith(
                '.mlpackage'):
            raise ValueError('extension should be `.mlpackage` when '
                             'convert_to == `mlprogram`. ')
        if self.convert_to == 'neuralnetwork' and not self.file_name.endswith(
                '.mlmodel'):
            raise ValueError('extension should be `.mlmodel` when '
                             'convert_to == `neuralnetwork`. ')

        super().check_param()


_BackendParam = CoreMLParam


@BACKEND_MANAGERS.register(
    'coreml', param=_BackendParam, ir_param=TorchScriptParam)
class CoreMLManager(BaseBackendManager):

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        return importlib.util.find_spec('coremltools') is not None

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('coremltools').version
            except Exception:
                return 'None'

    @classmethod
    def to_backend(cls,
                   torchscript_model: str,
                   output_path: str,
                   input_names: Sequence[str],
                   output_names: Sequence[str],
                   input_shapes: Dict[str, Dict],
                   min_shapes: Dict[str, Dict] = None,
                   max_shapes: Dict[str, Dict] = None,
                   compute_precision: str = 'FLOAT32',
                   convert_to: str = None,
                   minimum_deployment_target: Optional[str] = None,
                   skip_model_load: bool = True) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            torchscript_model (Union[str, torch.jit.RecursiveScriptModule]):
                The torchscript model to be converted.
            output_path (str): The output file.
            input_names (Sequence[str]): The input names of the model.
            output_names (Sequence[str]): The output names of the model.
            input_shapes (ShapeType): The Default shape of the inputs.
            min_shapes (ShapeType): The minimal shape of the inputs.
            max_shapes (ShapeType): The maximal shape of the inputs.
            compute_precision (str): The model precision, FLOAT16 or FLOAT32,
                read coremltools.precision for more detail, default `FLOAT32`.
            convert_to (str): The converted model type, can be
                'neuralnetwork' or 'mlprogram'. Defaults to 'neuralnetwork'.
            minimum_deployment_target (str, optional): minimum deploy target.
                iOS15, iOS16, etc., see coremltools.target
            skip_model_load (bool, optional): Skip model load.
                Defaults to True.
        Returns:
            Sequence[str]: Backend files.
        """
        from .torchscript2coreml import from_torchscript

        from_torchscript(
            torchscript_model,
            output_path,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes,
            min_shapes=min_shapes,
            max_shapes=max_shapes,
            compute_precision=compute_precision,
            convert_to=convert_to,
            minimum_deployment_target=minimum_deployment_target,
            skip_model_load=skip_model_load)

    @classmethod
    def to_backend_from_param(cls, ir_model: str, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            ir_model (str): The ir model path to perform the export.
            param (BaseBackendParam): Packed backend parameter.
        """
        param.check_param()

        assert isinstance(param, _BackendParam)
        assert isinstance(param.work_dir, str)
        assert isinstance(param.file_name, str)
        model_path = param.get_model_files()

        minimum_deployment_target = param.minimum_deployment_target
        cls.to_backend(
            ir_model,
            model_path,
            input_names=param.input_names,
            output_names=param.output_names,
            input_shapes=param.input_shapes,
            min_shapes=param.min_shapes,
            max_shapes=param.max_shapes,
            compute_precision=param.compute_precision,
            convert_to=param.convert_to,
            minimum_deployment_target=minimum_deployment_target,
            skip_model_load=param.skip_model_load)

    @classmethod
    def build_wrapper(cls, model_path: str):
        """Build the wrapper for the backend model.

        Args:
            model_path (str): Backend files.
        """
        from .wrapper import CoreMLWrapper
        return CoreMLWrapper(model_file=model_path)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        model_path = param.get_model_files()
        return cls.build_wrapper(model_path)

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
                                    get_model_inputs, load_config)

        deploy_cfg = config
        deploy_cfg = load_config(deploy_cfg)[0]

        common_params = get_common_config(deploy_cfg)
        model_params = get_model_inputs(deploy_cfg)[0]

        final_params = common_params
        final_params.update(model_params)

        ir_config = get_ir_config(deploy_cfg)
        input_names = ir_config.get('input_names', [])
        output_names = ir_config.get('output_names', [])
        input_shapes = final_params['input_shapes']
        min_shapes = dict(
            (name, shape['min_shape']) for name, shape in input_shapes.items())
        max_shapes = dict(
            (name, shape['max_shape']) for name, shape in input_shapes.items())
        input_shapes = dict((name, shape['default_shape'])
                            for name, shape in input_shapes.items())
        compute_precision = final_params.get('compute_precision', 'FLOAT32')
        convert_to = deploy_cfg.backend_config.convert_to

        minimum_deployment_target = final_params.get(
            'minimum_deployment_target', None)
        skip_model_load = final_params.get('skip_model_load', False)

        kwargs.setdefault('work_dir', work_dir)
        kwargs.setdefault('input_shapes', input_shapes)
        kwargs.setdefault('min_shapes', min_shapes)
        kwargs.setdefault('max_shapes', max_shapes)
        kwargs.setdefault('input_names', input_names)
        kwargs.setdefault('output_names', output_names)
        kwargs.setdefault('compute_precision', compute_precision)
        kwargs.setdefault('convert_to', convert_to)
        kwargs.setdefault('minimum_deployment_target',
                          minimum_deployment_target)
        kwargs.setdefault('skip_model_load', skip_model_load)

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
            name='convert', help='convert model from torchscript model.')
        export_parser.add_argument(
            '--torchscript-path',
            required=True,
            help='torchscript model path.')
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
                input_names=parsed_args.input_names,
                output_names=parsed_args.output_names,
                input_shapes=parsed_args.input_shapes,
                min_shapes=parsed_args.min_shapes,
                max_shapes=parsed_args.max_shapes,
                compute_precision=parsed_args.compute_precision,
                convert_to=parsed_args.convert_to,
                minimum_deployment_target=parsed_args.
                minimum_deployment_target,
                skip_model_load=parsed_args.skip_model_load)

            cls.to_backend_from_param(parsed_args.torchscript_path, param)
