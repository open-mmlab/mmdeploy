# Copyright (c) OpenMMLab. All rights reserved.

import logging
import os.path as osp
from typing import Any, Optional, Sequence

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('tvm')
class TVMManager(BaseBackendManager):

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
        from .wrapper import TVMWrapper
        bytecode = None if len(backend_files) <= 1 else backend_files[1]
        return TVMWrapper(
            backend_files[0],
            bytecode=bytecode,
            output_names=output_names,
            device=device)

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
                be save in this directory.
            deploy_cfg (Any): The deploy config.
            log_level (int, optional): The log level. Defaults to logging.INFO.
            device (str, optional): The device type. Defaults to 'cpu'.

        Returns:
            Seqeuence[str]: Backend files.
        """

        import copy

        from mmdeploy.apis.tvm import get_library_ext
        from mmdeploy.utils import (get_calib_filename, get_model_inputs,
                                    get_partition_config)
        from .onnx2tvm import from_onnx
        model_inputs = get_model_inputs(deploy_cfg)

        if device.startswith('cuda'):
            target = 'cuda'
        else:
            target = 'llvm'

        lib_ext = get_library_ext()

        tvm_files = []
        for model_id, onnx_path in enumerate(ir_files):
            model_input = copy.deepcopy(model_inputs[model_id])
            use_vm = model_input.get('use_vm', False)
            if 'target' not in model_input['tuner']:
                model_input['tuner']['target'] = target
            lib_path = osp.splitext(onnx_path)[0] + lib_ext
            code_path = osp.splitext(
                onnx_path)[0] + '.code' if use_vm else None
            model_input['output_file'] = lib_path
            model_input['onnx_model'] = onnx_path
            model_input['bytecode_file'] = code_path

            # create calibration dataset
            if 'qconfig' in model_input:
                from .quantize import HDF5Dataset
                calib_filename = get_calib_filename(deploy_cfg)
                calib_path = osp.join(work_dir, calib_filename)
                partition_cfgs = get_partition_config(deploy_cfg)
                onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
                partition_type = 'end2end' if partition_cfgs is None \
                    else onnx_name
                dataset = HDF5Dataset(
                    calib_path,
                    model_input['shape'],
                    model_type=partition_type,
                    device=target)
                model_input['dataset'] = dataset()

            from_onnx(**model_input)

            tvm_files += [lib_path, code_path]

        return tvm_files
