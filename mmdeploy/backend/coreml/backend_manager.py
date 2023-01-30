# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
from typing import Any, Optional, Sequence

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('coreml')
class CoreMLManager(BaseBackendManager):

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
        from .wrapper import CoreMLWrapper
        return CoreMLWrapper(model_file=backend_files[0])

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
        from mmdeploy.utils import (get_common_config, get_ir_config,
                                    get_model_inputs, load_config)
        from .torchscript2coreml import from_torchscript, get_model_suffix

        coreml_files = []
        for model_id, torchscript_path in enumerate(ir_files):
            torchscript_name = osp.splitext(osp.split(torchscript_path)[1])[0]
            output_file_prefix = osp.join(work_dir, torchscript_name)

            deploy_cfg = load_config(deploy_cfg)[0]

            common_params = get_common_config(deploy_cfg)
            model_params = get_model_inputs(deploy_cfg)[model_id]

            final_params = common_params
            final_params.update(model_params)

            ir_config = get_ir_config(deploy_cfg)
            input_names = ir_config.get('input_names', [])
            output_names = ir_config.get('output_names', [])
            input_shapes = final_params['input_shapes']
            compute_precision = final_params.get('compute_precision',
                                                 'FLOAT32')
            convert_to = deploy_cfg.backend_config.convert_to

            minimum_deployment_target = final_params.get(
                'minimum_deployment_target', None)
            skip_model_load = final_params.get('skip_model_load', False)
            from_torchscript(
                torchscript_path,
                output_file_prefix,
                input_names=input_names,
                output_names=output_names,
                input_shapes=input_shapes,
                compute_precision=compute_precision,
                convert_to=convert_to,
                minimum_deployment_target=minimum_deployment_target,
                skip_model_load=skip_model_load)

            suffix = get_model_suffix(convert_to)
            output_path = output_file_prefix + suffix
            coreml_files.append(output_path)
        return coreml_files
