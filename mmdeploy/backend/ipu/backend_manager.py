# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Optional, Sequence

from mmdeploy.utils import get_backend_config
from ..base import BACKEND_MANAGERS, BaseBackendManager
from mmdeploy.apis.ipu import onnx_to_popef


@BACKEND_MANAGERS.register('ipu')
class IPUManager(BaseBackendManager):

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
        from .wrapper import IPUWrapper

        # For unittest deploy_config will not pass into _build_wrapper
        # function.
        if deploy_cfg:
            backend_config = get_backend_config(deploy_cfg)
            bps = backend_config.get('batches_per_step', 1)
        else:
            bps = 1
        return IPUWrapper(
            popef_file=backend_files[0],
            bps=bps,
            output_names=output_names
        )

    @classmethod
    def to_backend(cls,
                   ir_files: Sequence[str],
                   work_dir: str,
                   deploy_cfg: Any,
                   log_level: int = logging.INFO,
                   device: str = 'cpu',
                   **kwargs) -> Sequence[str]:

        backend_files = []
        for model_id, onnx_path in enumerate(ir_files):
            model_name = onnx_path.split('/')[-1][:-5]
            ipu_config = deploy_cfg.get('backend_config', {})
            output_dir = ipu_config.get('output_dir', '')
            # assert output_dir != '', 'output dir for ipu backend is not set'
            # assert os.path.exists(output_dir), 'output dir not exist'

            if output_dir == '':
                output_dir = workdir

            model_dir = os.path.join(output_dir, model_name)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            ipu_config['output_dir'] = model_dir
            onnx_to_popef(onnx_path, ipu_config)
            backend_files.append(os.path.join(model_dir, 'executable.popef'))

        return backend_files
