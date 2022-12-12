# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Optional, Sequence

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('tensorrt')
class TensorRTManager(BaseBackendManager):

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

        from .wrapper import TRTWrapper
        return TRTWrapper(engine=backend_files[0], output_names=output_names)

    @classmethod
    def to_backend(cls,
                   ir_files: Sequence[str],
                   deploy_cfg: Any,
                   work_dir: str,
                   log_level: int = 20,
                   device: str = 'cpu',
                   **kwargs) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            ir_files (Sequence[str]): The intermediate representation files.
            deploy_cfg (Any): The deploy config.
            work_dir (str): The work directory, backend files and logs should
                be save in this directory.
            log_level (int, optional): The log level. Defaults to logging.INFO.
            device (str, optional): The device type. Defaults to 'cpu'.

        Returns:
            Seqeuence[str]: Backend files.
        """
        import os.path as osp

        from mmdeploy.utils import get_model_inputs, get_partition_config
        model_params = get_model_inputs(deploy_cfg)
        partition_cfgs = get_partition_config(deploy_cfg)
        assert len(model_params) == len(ir_files)

        from . import is_available
        assert is_available(), (
            'TensorRT is not available,'
            ' please install TensorRT and build TensorRT custom ops first.')

        from .onnx2tensorrt import onnx2tensorrt
        backend_files = []
        for model_id, model_param, onnx_path in zip(
                range(len(ir_files)), model_params, ir_files):
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            save_file = model_param.get('save_file', onnx_name + '.engine')

            partition_type = 'end2end' if partition_cfgs is None \
                else onnx_name
            onnx2tensorrt(
                work_dir,
                save_file,
                model_id,
                deploy_cfg,
                onnx_path,
                device=device,
                partition_type=partition_type)

            backend_files.append(osp.join(work_dir, save_file))

        return backend_files
