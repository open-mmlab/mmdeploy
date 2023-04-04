# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from mmdeploy.ir.torchscript import TorchScriptParam
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    FileNameDescriptor)


# We name the name `TorchJIT` to distinguish `Torchscript` as IR.
@dataclass
class TorchJITParam(BaseBackendParam):
    """TorchJIT backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        input_names (List[str]): Names of the inputs.
        output_names (List[str]): Names of the outputs.
    """
    file_name: FileNameDescriptor = FileNameDescriptor(
        default=None, postfix='.pth')

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str), ('Expect string work_dir, '
                                                f'got {self.work_dir}')
        assert isinstance(self.file_name, str), ('Expect string file_name, '
                                                 f'got {self.file_name}')
        return osp.join(self.work_dir, self.file_name)


_BackendParam = TorchJITParam


@BACKEND_MANAGERS.register(
    'torchscript', param=_BackendParam, ir_param=TorchScriptParam)
class TorchScriptManager(BaseBackendManager):

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        ret = importlib.util.find_spec('torch') is not None

        if ret and with_custom_ops:
            from .init_plugins import ops_available
            ret = ret and ops_available()

        return ret

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('torch').version
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
            ops_info = f'torchscript custom ops:\t{ops_available}'
            log_callback(ops_info)
            info = f'{info}\n{ops_info}'

        return info

    @classmethod
    def to_backend(cls, torhscript_path: str, save_path: str):
        """Convert intermediate representation to given backend.

        Args:
            torhscript_path (str): The intermediate representation files.
            save_path (str): The save path of onnx path.
        Returns:
            Sequence[str]: Backend files.
        """
        if osp.abspath(save_path) != osp.abspath(torhscript_path):
            shutil.copy(torhscript_path, save_path)

    @classmethod
    def to_backend_from_param(cls, ir_model: str, param: BaseBackendParam):
        """Export to backend with packed backend parameter.

        Args:
            ir_model (str): The ir model path to perform the export.
            param (BaseBackendParam): Packed backend parameter.
        """
        assert isinstance(param.work_dir, str)
        assert isinstance(param.file_name, str)
        save_path = osp.join(param.work_dir, param.file_name)
        cls.to_backend(ir_model, save_path)

    @classmethod
    def build_wrapper(cls,
                      model_path: str,
                      input_names: Optional[Sequence[str]] = None,
                      output_names: Optional[Sequence[str]] = None):
        """Build the wrapper for the backend model.

        Args:
            model_path (str): torchscript model path.
            input_names (Optional[Sequence[str]], optional): input names.
                Defaults to None.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
        """
        from .wrapper import TorchscriptWrapper
        return TorchscriptWrapper(
            model=model_path,
            input_names=input_names,
            output_names=output_names)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (_BackendParam): Packed backend parameter.
        """
        assert isinstance(param, _BackendParam)
        assert isinstance(param.work_dir, str)
        assert isinstance(param.file_name, str)
        model_path = osp.join(param.work_dir, param.file_name)
        input_names = param.input_names
        output_names = param.output_names
        return cls.build_wrapper(
            model_path, input_names=input_names, output_names=output_names)

    @classmethod
    def build_param_from_config(cls,
                                config: Any,
                                work_dir: str,
                                backend_files: List[str] = None,
                                **kwargs) -> _BackendParam:
        """Build param from deploy config.

        Args:
            config (Any): The deploy config.
            work_dir (str): work directory of the parameters.
            backend_files (List[str]): The backend files of the model.

        Returns:
            BaseBackendParam: The packed backend parameter.
        """
        from mmdeploy.utils import get_ir_config
        ir_config = get_ir_config(config)
        input_names = ir_config.get('input_names', [])
        output_names = ir_config.get('output_names', [])
        kwargs.update(dict(input_names=input_names, output_names=output_names))
        return _BackendParam(
            work_dir=work_dir, file_name=backend_files[0], **kwargs)
