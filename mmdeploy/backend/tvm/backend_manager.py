# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import re
import sys
from argparse import Action, ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from mmdeploy.ir.onnx import ONNXParam
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    dataclass_property, get_obj_by_qualname,
                    import_custom_modules)


def get_library_ext() -> str:
    """Get the extension of the library.

    Returns:
        str: The extension name
    """
    platform = sys.platform.lower()
    if platform == 'win32' or platform == 'cygwin':
        return '.dll'
    elif platform == 'linux' or platform == 'darwin' or platform == 'freebsd':
        return '.so'


class DTypeAction(Action):
    """dtype argparse action."""

    def __call__(self, parser, namespace, values, option_string=None):
        """call action."""
        args = values
        if isinstance(args, str):
            args = [args]

        pattern = r'^[^\S\n]*((?P<input_name>\w+):)?(?P<dtype>\w+)$'
        ret = dict()
        for arg in args:
            arg = [i.strip() for i in arg.split(',')]
            for single_arg in arg:
                if len(single_arg) == 0:
                    continue
                m = re.match(pattern, single_arg)
                if m is None:
                    raise ValueError(f'Can not parse shape: {single_arg}')
                input_name = m.group('input_name')
                dtype = m.group('dtype')
                if input_name in ret:
                    raise NameError(
                        f'shape of `{input_name}` has been assigned'
                        'more than once.')
                ret[input_name] = dtype

        setattr(namespace, self.dest, ret)


@dataclass
class TVMParam(BaseBackendParam):
    """TVM backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        vm_name (str): Serialized vm file. If not given,
            vm_name would be he same as file_name with postfix `.vm`
        use_vm (bool): Enable tvm virtual machine runtime. Defaults to False.
        input_shapes (ShapeType): The Default shape of the inputs.
        output_names (List[str]): Names of the outputs.
        dtypes (Dict[str, str]): The input data types.
        tuner (Optional[Union[TVMTunerBase, Dict]], optional): The tuner
            config. Defaults to None.
        qconfig (QConfig): `relay.quantize.QConfig` instance.
        quanti_data (Any): Calibration dataset. Iterable object of
            `Dict[str, ndarray]`
        device (str): Device used to perform inference.
    """
    _default_postfix = get_library_ext()
    _vm_postfix = '.vm'
    _vm_name = None

    vm_name: str = None
    use_vm: bool = False
    dtypes: Dict[str, str] = None
    tuner: Any = None
    qconfig: Any = None
    device: str = 'llvm'

    @dataclass_property
    def vm_name(self) -> str:
        """vm_name getter."""
        if self._vm_name is None and self.file_name is not None:
            # if bin name has not been given, use file name with postfix
            name = osp.splitext(self.file_name)[0]
            return name + self._vm_postfix
        return self._vm_name

    @vm_name.setter
    def vm_name(self, val) -> None:
        """vm_name setter."""
        if val is not None and osp.splitext(val)[1] == '':
            val = val + self._vm_postfix

        self._vm_name = val

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str)
        assert isinstance(self.file_name, str)
        param_file_path = osp.join(self.work_dir, self.file_name)
        assert isinstance(self.vm_name, str)
        algorithm_file_path = osp.join(self.work_dir, self.vm_name)
        return param_file_path, algorithm_file_path

    @classmethod
    def add_argument(cls, parser: ArgumentParser, name: str, dtype: Any,
                     default: Any, desc: str):
        arg_name = f'--{name.replace("_", "-")}'
        if name == 'dtypes':
            parser.add_argument(
                arg_name, action=DTypeAction, nargs='+', help=desc)
        else:
            return super().add_argument(parser, name, dtype, default, desc)


_BackendParam = TVMParam


@BACKEND_MANAGERS.register('tvm', param=TVMParam, ir_param=ONNXParam)
class TVMManager(BaseBackendManager):

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        ret = importlib.util.find_spec('tvm') is not None

        return ret

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('tvm').version
            except Exception:
                return 'None'

    @classmethod
    def to_backend(cls,
                   onnx_file: str,
                   output_file: str,
                   use_vm: bool = False,
                   vm_file: str = '',
                   input_shapes: Optional[Dict] = None,
                   dtypes: Union[str, Dict] = 'float32',
                   tuner: Optional[Union[Any, Dict, str]] = None,
                   qconfig: Optional[Union[Any, Dict, str]] = None,
                   dataset: Optional[Iterable] = None,
                   device: str = 'llvm') -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            onnx_file (str): The intermediate representation files.
            output_file (str): output library path
            use_vm (bool, optional): Enable tvm virtual machine runtime.
                Defaults to False.
            vm_file (str, optional): output bytecode path for virtual
                machine. Defaults to ''.
            input_shapes (Optional[Dict], optional): The input shape
                dictionary. Defaults to None.
            dtypes (Union[str, Dict], optional): The input data type
                dictionary. Defaults to 'float32'.
            tuner (Optional[Union[TVMTunerBase, Dict]], optional): The tuner
                config. Defaults to None.
            qconfig (QConfig): `relay.quantize.QConfig` instance.
            dataset (Any): Calibration dataset. Iterable object of
                `Dict[str, ndarray]`
            device (str): Device used to perform inference.
        Returns:
            Sequence[str]: Backend files.
        """
        from .onnx2tvm import from_onnx

        # process dtypes
        if isinstance(dtypes, Dict) and len(dtypes) == 1 and None in dtypes:
            dtypes = dtypes[None]

        # process tuner
        if isinstance(tuner, str):
            tuner = get_obj_by_qualname(tuner)

        # process qconfig
        if isinstance(qconfig, str):
            qconfig = get_obj_by_qualname(qconfig)

        # process device
        if device.startswith('cuda'):
            device = 'cuda'
        else:
            device = 'llvm'

        from_onnx(
            onnx_file,
            output_file=output_file,
            use_vm=use_vm,
            bytecode_file=vm_file,
            shape=input_shapes,
            dtype=dtypes,
            tuner=tuner,
            qconfig=qconfig,
            dataset=dataset,
            device=device)

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
        assert isinstance(param.vm_name, str)
        vm_path = osp.join(param.work_dir, param.vm_name)

        cls.to_backend(
            ir_model,
            model_path,
            use_vm=param.use_vm,
            vm_file=vm_path,
            input_shapes=param.input_shapes,
            dtypes=param.dtypes,
            tuner=param.tuner,
            qconfig=param.qconfig,
            dataset=param.quanti_data,
            device=param.device)

    @classmethod
    def build_wrapper(
        cls,
        lib_file: str,
        output_names: Optional[Sequence[str]],
        vm_file: str = None,
        device: str = 'cpu',
    ):
        """Build the wrapper for the backend model.

        Args:
            lib_file (str): generated library path
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
            vm_file (str, optional): output bytecode path for virtual
                machine. Defaults to ''.
            device (str, optional): The device info. Defaults to 'cpu'.
        """
        from .wrapper import TVMWrapper
        if vm_file is not None and not osp.exists(vm_file):
            vm_file = None
        return TVMWrapper(
            lib_file,
            output_names=output_names,
            bytecode=vm_file,
            device=device)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        model_path, vm_path = param.get_model_files()
        output_names = param.output_names
        device = param.device
        return cls.build_wrapper(
            model_path,
            output_names=output_names,
            vm_file=vm_path,
            device=device)

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
        from mmdeploy.utils import (get_calib_config, get_ir_config,
                                    get_model_inputs)

        # get output names
        ir_config = get_ir_config(config)
        output_names = ir_config['output_names']
        kwargs.setdefault('output_names', output_names)

        # get tvm param
        model_inputs = get_model_inputs(config)
        shape = model_inputs[0]['shape']
        dtype = model_inputs[0].get('dtype', 'float32')
        tuner = model_inputs[0].get('tuner', dict(type='DefaultTuner'))

        kwargs.setdefault('work_dir', work_dir)
        kwargs.setdefault('input_shapes', shape)
        kwargs.setdefault('dtypes', dtype)
        kwargs.setdefault('tuner', tuner)

        qconfig = model_inputs[0].get('qconfig', None)
        if qconfig is not None:
            from ..base import create_h5pydata_generator
            kwargs.setdefault('qconfig', qconfig)
            calib_config = get_calib_config(config)
            assert 'calib_file' in calib_config
            calib_path = osp.join(work_dir, calib_config['calib_file'])
            kwargs.setdefault('quanti_data',
                              create_h5pydata_generator(calib_path, shape))

        backend_files = [] if backend_files is None else backend_files
        if len(backend_files) > 0:
            kwargs['file_name'] = backend_files[0]
        if len(backend_files) > 1:
            kwargs['vm_name'] = backend_files[1]
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
                vm_name=parsed_args.vm_name,
                use_vm=parsed_args.use_vm,
                input_shapes=parsed_args.input_shapes,
                output_names=parsed_args.output_names,
                dtypes=parsed_args.dtypes,
                tuner=parsed_args.tuner,
                qconfig=parsed_args.qconfig,
                quanti_data=parsed_args.quanti_data,
                device=parsed_args.device)

            cls.to_backend_from_param(parsed_args.onnx_path, param)
