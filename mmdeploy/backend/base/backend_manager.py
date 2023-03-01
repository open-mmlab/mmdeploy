# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import logging
import os.path as osp
import re
from abc import ABCMeta
from argparse import Action, ArgumentParser
from collections import OrderedDict
from dataclasses import MISSING, dataclass, field, fields
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Union)

from mmdeploy.utils.docstring_parser import inspect_docstring_arguments


def _parse_shape_type(args: Union[str, List[str]],
                      allow_placeholder: bool = True) -> Dict[str, List]:
    """parse the shape of the arguments.

    Args:
        args (Union[str, List[str]]): The arguments string to be parsed.
        allow_placeholder (bool, optional): Allow non-int shape.

    Returns:
        Dict[str, List]: The parsed shapes
    """
    if isinstance(args, str):
        args = [args]

    if allow_placeholder:
        pattern = r'^[^\S\n]*(((?P<input_name>\w+):)?(?P<shape>(\?|\d+)(x(\?|\d+))*))+$'  # noqa
    else:
        pattern = r'^[^\S\n]*(((?P<input_name>\w+):)?(?P<shape>\d+(x\d+)*))+$'  # noqa

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
            shape = [
                None if i == '?' else int(i)
                for i in m.group('shape').split('x')
            ]
            if input_name in ret:
                raise NameError(f'shape of `{input_name}` has been assigned'
                                'more than once.')
            ret[input_name] = shape

    return ret


class ShapeTypeAction(Action):
    """Shape type argparse action."""

    def __call__(self, parser, namespace, values, option_string=None):
        """call action."""
        ret = _parse_shape_type(values, allow_placeholder=False)
        setattr(namespace, self.dest, ret)


class ShapePlaceHolderAction(Action):
    """Shape type argparse action with question mark placeholder."""

    def __call__(self, parser, namespace, values, option_string=None):
        """call action."""
        ret = _parse_shape_type(values, allow_placeholder=True)
        setattr(namespace, self.dest, ret)


ShapeType = Dict[str, Sequence[int]]


class dataclass_property(property):

    def __set__(self, obj, value):
        if isinstance(value, property):
            # dataclasses tries to set a default and uses the
            # getattr(cls, name). But the real default will come
            # from: `_attr = field(..., default=...)`.
            return
        super().__set__(obj, value)


@dataclass
class BaseBackendParam:
    """Base backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        input_shapes (ShapeType): The Default shape of the inputs.
        min_shapes (ShapeType): The minimal shape of the inputs.
        max_shapes (ShapeType): The maximal shape of the inputs.
        input_names (List[str]): Names of the inputs.
        output_names (List[str]): Names of the outputs.
        device (str): Device used to perform inference.
        quanti_data (Union[Iterable, str]): Iterable object to provide the
            quantization data. Each iteration gives a dict of input name and
            correspond tensor.
        uri (str): The uri of remote device.
    """
    _default_postfix = ''
    _file_name = None
    _manager = None

    work_dir: str = None
    file_name: str
    min_shapes: ShapeType = field(default_factory=OrderedDict)
    input_shapes: ShapeType = field(default_factory=OrderedDict)
    max_shapes: ShapeType = field(default_factory=OrderedDict)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    device: str = 'cpu'
    quanti_data: Union[Iterable, str] = None
    uri: str = None

    @dataclass_property
    def file_name(self) -> str:
        """file_name getter."""
        return self._file_name

    @file_name.setter
    def file_name(self, val) -> None:
        """file_name setter."""
        if val is not None and osp.splitext(val)[1] == '':
            val = val + self._default_postfix

        self._file_name = val

    def get_model_files(self) -> Union[str, List[str]]:
        """get model files."""
        raise NotImplementedError(
            f'get_model_files has not implemented for {type(self).__name__}')

    def fix_param(self):
        """Fix shapes and names in the parameter."""

        def _fix_none_name(shapes):
            if shapes is not None and None in shapes:
                if len(shapes) != 1 or len(
                        self.input_names) != 1 or self.input_names[0] is None:
                    raise ValueError(
                        f'Can not inference name with shapes: {shapes}'
                        f' and input_names: {self.input_names}')
                return {self.input_names[0]: shapes[None]}
            else:
                return shapes

        def _fill_shape_placeholder(default_shape, shape):
            if len(shape) != len(default_shape):
                raise ValueError(
                    f'Can not fill placeholder {shape} with {default_shape}')
            return [
                a if a is not None else b
                for a, b in zip(shape, default_shape)
            ]

        if not isinstance(self.input_shapes, Dict):
            raise TypeError('Expect dict input shapes,'
                            f' but got {type(self.input_shapes)}')

        # fill input names
        if len(self.input_names) == 0:
            self.input_names = list(self.input_shapes.keys())

        if len(self.input_names) != len(set(self.input_names)):
            raise ValueError(
                f'Duplicate names in input_names: {self.input_names}')

        if None in self.input_names:
            raise ValueError('Found None in input names.')

        # fix input shapes with no names
        self.input_shapes = _fix_none_name(self.input_shapes)

        if self.min_shapes is None or len(self.min_shapes) == 0:
            self.min_shapes = self.input_shapes
        if self.max_shapes is None or len(self.max_shapes) == 0:
            self.max_shapes = self.input_shapes

        if not isinstance(self.min_shapes, Dict):
            raise TypeError(
                f'Expect min shapes type Dict, got {type(self.min_shapes)}.')
        if not isinstance(self.max_shapes, Dict):
            raise TypeError(
                f'Expect max shapes type Dict, got {type(self.max_shapes)}.')

        self.min_shapes = _fix_none_name(self.min_shapes)
        self.max_shapes = _fix_none_name(self.max_shapes)

        # fix placeholder min/max shapes
        for name, in_shape in self.input_shapes.items():
            if name in self.min_shapes:
                self.min_shapes[name] = _fill_shape_placeholder(
                    in_shape, self.min_shapes[name])

            if name in self.max_shapes:
                self.max_shapes[name] = _fill_shape_placeholder(
                    in_shape, self.max_shapes[name])

    def check_param(self):
        """Check the parameter validation."""
        self.fix_param()

        input_shapes = self.input_shapes
        min_shapes = self.min_shapes
        max_shapes = self.max_shapes

        if not (len(input_shapes) == len(min_shapes) == len(max_shapes)):
            raise ValueError(f'len(min_shapes) = {len(min_shapes)}\n',
                             f'len(input_shapes) = {len(input_shapes)}\n',
                             f'len(max_shapes) = {len(max_shapes)}\n',
                             ' should be the same.')

        for name, input_shape in input_shapes.items():
            if name not in min_shapes:
                raise NameError(f'{name} not found in min_shapes.')
            if name not in max_shapes:
                raise NameError(f'{name} not found in max_shapes.')
            min_shape = min_shapes[name]
            max_shape = max_shapes[name]

            if not isinstance(input_shape, Sequence):
                raise TypeError(f'input shape of {name} is not sequence.')
            if not isinstance(min_shape, Sequence):
                raise TypeError(f'min shape of {name} is not sequence.')
            if not isinstance(max_shape, Sequence):
                raise TypeError(f'max shape of {name} is not sequence.')
            if not (len(input_shape) == len(min_shape) == len(max_shape)):
                raise ValueError(
                    f'len(min_shapes[{name}]) = {len(min_shape)}\n',
                    f'len(input_shapes[{name}]) = {len(input_shape)}\n',
                    f'len(max_shapes[{name}]) = {len(max_shape)}\n',
                    ' should be the same.')

            for min_s, opt_s, max_s in zip(min_shape, input_shape, max_shape):
                if not min_s <= opt_s <= max_s:
                    raise ValueError(
                        f'Input {name} has invalid shape:\n',
                        f'min shape:{min_shape}\n',
                        f'input shape:{input_shape}\n',
                        f'max shape:{max_shape}',
                    )

    @classmethod
    def get_manager(cls):
        """Get backend manager."""
        return cls._manager

    @classmethod
    def add_argument(cls, parser: ArgumentParser, name: str, dtype: Any,
                     default: Any, desc: str):
        """Add argument to the parser.

        Args:
            parser (ArgumentParser): Parser object to add argument.
            name (str): Name of the argument.
            dtype (Any): Argument type.
            default (Any): Default value of the argument.
            desc (str): Description of the argument.
        """
        arg_name = f'--{name.replace("_", "-")}'
        if dtype == bool:
            if default is True:
                action = 'store_false'
            else:
                action = 'store_true'
            parser.add_argument(arg_name, action=action, help=desc)
        elif dtype == ShapeType:
            action = ShapeTypeAction \
                if name == 'input_shapes' else ShapePlaceHolderAction
            parser.add_argument(
                arg_name, action=action, nargs='+', default=default, help=desc)
        elif dtype == List[str]:
            parser.add_argument(
                arg_name, type=str, nargs='+', default=default, help=desc)
        else:
            parser.add_argument(
                arg_name, type=dtype, default=default, help=desc)

    @classmethod
    def add_arguments(cls,
                      parser: ArgumentParser,
                      ignore_fields: Optional[List] = None):
        """Add Arguments to the parser to build the param.

        Args:
            parser (ArgumentParser): Parser to add arguments.
            ignore_fields (Optional[List], optional): Ignore some fields in
                the dataclass. Defaults to None.
        """
        parser.description = f'build {cls.__name__}'

        if ignore_fields is None:
            ignore_fields = []
        remain_fields = inspect_docstring_arguments(
            cls, ignore_args=ignore_fields)

        field_map = dict((f.name, f) for f in fields(cls))

        for remain_field in remain_fields:
            name = remain_field.name
            desc = remain_field.desc

            assert name in field_map, \
                f'{name} is not a field in {cls.__name__}'
            cls_field = field_map[name]

            dtype = cls_field.type
            if cls_field.default is not MISSING:
                default = cls_field.default
            elif isinstance(cls_field.default_factory, Callable):
                default = cls_field.default_factory()
            else:
                default = None

            cls.add_argument(
                parser, name, dtype=dtype, default=default, desc=desc)


class BaseBackendManager(metaclass=ABCMeta):
    """Abstract interface of backend manager."""

    build_ir_param = None
    build_param = BaseBackendParam

    @classmethod
    def build_wrapper(cls, *args, **kwargs):
        """Build the wrapper for the backend model."""
        raise NotImplementedError(
            f'build_wrapper has not been implemented for `{cls.__name__}`')

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        raise NotImplementedError(
            f'is_available has not been implemented for "{cls.__name__}"')

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        raise NotImplementedError(
            f'get_version has not been implemented for "{cls.__name__}"')

    @classmethod
    def check_env(cls, log_callback: Callable = lambda _: _) -> str:
        """Check current environment.

        Returns:
            str: Info about the environment.
        """
        try:
            available = cls.is_available()
            if available:
                try:
                    backend_version = cls.get_version()
                except NotImplementedError:
                    backend_version = 'Unknown'
            else:
                backend_version = 'None'

            info = f'{cls.backend_name}:\t{backend_version}'
        except Exception:
            info = f'{cls.backend_name}:\tCheckFailed'

        log_callback(info)

        return info

    @classmethod
    def to_backend(cls,
                   ir_files: Sequence[str],
                   work_dir: str,
                   deploy_cfg: Any,
                   log_level: int = logging.INFO,
                   device: str = 'cpu',
                   **kwargs) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            ir_files (Sequence[str]): The intermediate representation files.
            work_dir (str): The work directory, backend files and logs should
                be saved in this directory.
            deploy_cfg (Any): The deploy config.
            log_level (int, optional): The log level. Defaults to logging.INFO.
            device (str, optional): The device type. Defaults to 'cpu'.
        Returns:
            Sequence[str]: Backend files.
        """
        raise NotImplementedError(
            f'to_backend has not been implemented for `{cls.__name__}`')

    @classmethod
    def to_backend_from_param(cls, ir_model: str, param: BaseBackendParam):
        """Export to backend with packed backend parameter.

        Args:
            ir_model (str): The ir model path to perform the export.
            param (BaseBackendParam): Packed backend parameter.
        """
        raise NotImplementedError(
            'to_backend_from_param has not been implemented for '
            f'`{cls.__name__}`')

    @classmethod
    def build_wrapper_from_param(cls, param: BaseBackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        raise NotImplementedError(
            'build_wrapper_from_param has not been implemented for '
            f'`{cls.__name__}`')

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
        raise NotImplementedError(
            f'parse_args of {cls.__name__} has not been implemented.')

    @classmethod
    def main(cls):
        """Create console tools."""
        parser = ArgumentParser()
        generator = cls.parse_args(parser)

        try:
            while True:
                next(generator)
        except StopIteration:
            pass

    @classmethod
    def build_param_from_config(cls,
                                config: Any,
                                work_dir: str,
                                backend_files: List[str] = None,
                                **kwargs) -> BaseBackendParam:
        """Build param from deploy config.

        This is a bridge between old and new api.

        Args:
            config (Any): The deploy config.
            work_dir (str): work directory of the parameters.
            backend_files (List[str]): The backend files of the model.

        Returns:
            BaseBackendParam: The packed backend parameter.
        """
        raise NotImplementedError(
            'build_param_from_config has not been implemented'
            f' for {cls.__name__}.')


class BackendManagerRegistry:
    """backend manager registry."""

    def __init__(self):
        self._module_dict = {}

    def register(self,
                 name: str,
                 enum_name: Optional[str] = None,
                 param: Any = None,
                 ir_param: Any = None):
        """register backend manager.

        Args:
            name (str): name of the backend
            enum_name (Optional[str], optional): enum name of the backend.
                if not given, the upper case of name would be used.
        """
        from mmdeploy.utils import get_root_logger
        logger = get_root_logger()

        if enum_name is None:
            enum_name = name.upper()

        def wrap_manager(cls):

            from mmdeploy.utils import Backend

            if not hasattr(Backend, enum_name):
                from aenum import extend_enum
                extend_enum(Backend, enum_name, name)
                logger.info(f'Registry new backend: {enum_name} = {name}.')

            if name in self._module_dict:
                logger.info(
                    f'Backend manager of `{name}` has already been registered.'
                )

            self._module_dict[name] = cls

            cls.backend_name = name
            cls.build_param = param
            cls.build_ir_param = ir_param
            if param is not None:
                param._manager = cls

            return cls

        return wrap_manager

    def find(self, name: str) -> BaseBackendManager:
        """Find the backend manager with name.

        Args:
            name (str): backend name.
        Returns:
            BaseBackendManager: backend manager of the given backend.
        """
        # try import backend if backend is in `mmdeploy.backend`
        try:
            importlib.import_module('mmdeploy.backend.' + name)
        except Exception:
            pass
        return self._module_dict.get(name, None)


BACKEND_MANAGERS = BackendManagerRegistry()


def get_backend_manager(name: str) -> BaseBackendManager:
    """Get backend manager.

    Args:
        name (str): name of the backend.
    Returns:
        BaseBackendManager: The backend manager of given name
    """
    from enum import Enum
    if isinstance(name, Enum):
        name = name.value
    return BACKEND_MANAGERS.find(name)
