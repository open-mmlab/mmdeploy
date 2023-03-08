# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Union

from mmdeploy.ir.onnx import ONNXParam
from mmdeploy.utils import get_common_config, get_root_logger
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    get_obj_by_qualname)


@dataclass
class VACCParam(BaseBackendParam):
    """VACC backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        input_shapes (ShapeType): The Default shape of the inputs.
        input_names (List[str]): Names of the inputs.
        output_names (List[str]): Names of the outputs.
        qconfig (Dict): Dictionary arguments feed to vacc.qconfig.
            Or qualname to the dict.
        quanti_data (Any): Calibration dataset. Iterable object of
            `Dict[str, ndarray]`
    """
    _default_postfix = ''

    use_vm: bool = False
    qconfig: Union[str, Dict] = field(default_factory=dict)

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str)
        assert isinstance(self.file_name, str)
        if isinstance(self.qconfig, str):
            qconfig = get_obj_by_qualname(self.qconfig)
        else:
            qconfig = self.qconfig
            assert isinstance(qconfig, Dict)
        quant_mode = qconfig.get('dtype', 'fp16')
        save_dir = '-'.join([self.file_name, quant_mode])
        model_prefix = osp.join(self.work_dir, save_dir, self.file_name)
        return [
            model_prefix + '.so', model_prefix + '.json',
            model_prefix + '.params'
        ]


_BackendParam = VACCParam


@BACKEND_MANAGERS.register('vacc', param=VACCParam, ir_param=ONNXParam)
class VACCManager(BaseBackendManager):

    @classmethod
    def build_wrapper(cls,
                      backend_files: Sequence[str],
                      device: str = 'cpu',
                      input_names: Optional[Sequence[str]] = None,
                      output_names: Optional[Sequence[str]] = None,
                      deploy_cfg: Optional[Any] = None,
                      **kwargs):
        """Build the wrapper for the backend model.

        Args:
            backend_files (Sequence[str]): Backend files.
            device (str, optional): The device info. Defaults to 'cpu'.
            input_names (Optional[Sequence[str]], optional): input names.
                Defaults to None.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
            deploy_cfg (Optional[Any], optional): The deploy config. Defaults
                to None.
        """
        from .wrapper import VACCWrapper

        # For unittest deploy_config will not pass into _build_wrapper
        # function.

        try:
            common_cfg = get_common_config(deploy_cfg)
            vdsp_params_info = common_cfg['vdsp_params_info']

            return VACCWrapper(
                lib_file=backend_files[0],
                graph_file=backend_files[1],
                param_file=backend_files[2],
                vdsp_params_info=vdsp_params_info,
                output_names=output_names)
        except Exception:
            print('Build model process success, wrapper process stopped')
            exit(1)

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
        qconfig: Optional[Dict] = None,
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

        from_onnx(
            onnx_model,
            output_path=output_path,
            model_name=model_name,
            input_shapes=input_shapes,
            qconfig=qconfig)

        # model_inputs = get_model_inputs(deploy_cfg)
        # common_params = get_common_config(deploy_cfg)
        # model_name = common_params['name']

        # backend_files = []
        # for model_id, onnx_path in zip(range(len(ir_files)), ir_files):
        #     model_input = copy.deepcopy(model_inputs[model_id])
        #     model_file = from_onnx(onnx_path, work_dir, model_input,
        #                            model_name)
        #     backend_files += model_file

        # return backend_files

        # model_inputs = get_model_inputs(deploy_cfg)
        # common_params = get_common_config(deploy_cfg)
        # model_name = common_params['name']
