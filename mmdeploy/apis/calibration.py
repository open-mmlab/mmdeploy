# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Optional, Union

from mmengine import Config

from .core import PIPELINE_MANAGER, no_mp


@PIPELINE_MANAGER.register_pipeline()
def create_calib_input_data(calib_file: str,
                            deploy_cfg: Union[str, Config],
                            model_cfg: Union[str, Config],
                            model_checkpoint: Optional[str] = None,
                            dataset_cfg: Optional[Union[str, Config]] = None,
                            dataset_type: str = 'val',
                            device: str = 'cpu') -> None:
    """Create dataset for post-training quantization.

    Args:
        calib_file (str): The output calibration data file.
        deploy_cfg (str | Config): Deployment config file or
            Config object.
        model_cfg (str | Config): Model config file or Config object.
        model_checkpoint (str): A checkpoint path of PyTorch model,
            defaults to `None`.
        dataset_cfg (Optional[Union[str, Config]], optional): Model
            config to provide calibration dataset. If none, use `model_cfg`
            as the dataset config. Defaults to None.
        dataset_type (str, optional): The dataset type. Defaults to 'val'.
        device (str, optional): Device to create dataset. Defaults to 'cpu'.
    """

    from mmdeploy.core import patch_model
    from mmdeploy.utils import (IR, cfg_apply_marks, get_backend,
                                get_ir_config, load_config)
    from .utils import create_calib_input_data as create_calib_input_data_impl
    with no_mp():
        if dataset_cfg is None:
            dataset_cfg = model_cfg

        # load cfg if necessary
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        if dataset_cfg is None:
            dataset_cfg = model_cfg

        # load dataset_cfg if necessary
        dataset_cfg = load_config(dataset_cfg)[0]
        calib_dataloader = deepcopy(dataset_cfg[f'{dataset_type}_dataloader'])
        calib_dataloader['batch_size'] = 1

        from mmdeploy.apis.utils import build_task_processor
        task_processor = build_task_processor(model_cfg, deploy_cfg, device)

        apply_marks = cfg_apply_marks(deploy_cfg)

        model = task_processor.build_pytorch_model(model_checkpoint)
        dataset = task_processor.build_dataset(calib_dataloader['dataset'])
        calib_dataloader['dataset'] = dataset
        dataloader = task_processor.build_dataloader(calib_dataloader)

        # patch model
        backend = get_backend(deploy_cfg).value
        ir = IR.get(get_ir_config(deploy_cfg)['type'])
        patched_model = patch_model(
            model, cfg=deploy_cfg, backend=backend, ir=ir)

        def get_tensor_func(input_data):
            input_data = model.data_preprocessor(input_data)
            return input_data['inputs']

        create_calib_input_data_impl(
            calib_file,
            patched_model,
            dataloader,
            get_tensor_func=get_tensor_func,
            inference_func=model.forward,
            model_partition=apply_marks,
            context_info=dict(cfg=deploy_cfg),
            device=device)
