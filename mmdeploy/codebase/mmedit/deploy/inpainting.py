# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.parallel import collate, scatter
from mmcv.utils import Config
from torch.utils.data import Dataset

from mmdeploy.codebase.base import BaseTask
from mmdeploy.codebase.mmedit.deploy.mmediting import MMEDIT_TASK
from mmdeploy.utils import Task, get_input_shape, get_ir_config, load_config


def process_model_config(model_cfg: Config,
                         imgs: Union[Sequence[str], Sequence[np.ndarray]],
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (Config): The model config.
        imgs (Sequence[str] | Sequence[np.ndarray]): Input image(s), accepted
            data type are List[str], List[np.ndarray].
        input_shape (Sequence[int], optional): A list of two integer
            in (width, height) format specifying input shape. Default: None.

    Returns:
        Config: the model config after processing.
    """
    config = load_config(model_cfg)[0].copy()
    load_from_file = isinstance(imgs[0], str)
    if not load_from_file:
        # Remove 'LoadImageFromFile' and 'LoadMask'
        config.test_pipeline.pop(1)
        config.test_pipeline.pop(0)

    is_static_cfg = input_shape is not None
    if is_static_cfg:
        assert 'Resize' in [_.type for _ in config.test_pipeline]
        for pipeline in config.test_pipeline[::-1]:
            if pipeline.type == 'Resize':
                pipeline.scale = input_shape
                break

    key = 'gt_img_path'
    for pipeline in config.test_pipeline:
        if 'meta_keys' in pipeline:
            while key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)

    return config


@MMEDIT_TASK.register_module(Task.INPAINTING.value)
class Inpainting(BaseTask):
    """BaseTask class of inpainting task.

    Args:
        model_cfg (Config): Model config file.
        deploy_cfg (Config): Deployment config file.
        device (str): A string specifying device type.
    """

    def __init__(self, model_cfg: Config, deploy_cfg: Config, device: str):
        super(Inpainting, self).__init__(model_cfg, deploy_cfg, device)

    def init_backend_model(self,
                           model_files: Sequence[str] = None,
                           **kwargs) -> nn.Module:
        from .inpainting_model import build_inpainting_model

        return build_inpainting_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            **kwargs)

    def init_pytorch_model(self,
                           model_checkpoint: Optional[str] = None,
                           **kwargs) -> nn.Module:
        from mmedit.apis import init_model

        model = init_model(self.model_cfg, model_checkpoint, self.device)

        forward_test = model.forward_test
        model.forward_test = lambda *args, **kwargs: {
            k: v
            for k, v in forward_test(*args, **kwargs).items()
            if k in get_ir_config(self.deploy_cfg).output_names
        }

        return model.eval()

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Optional[Sequence[int]] = None,
                     pipeline_updater: Optional[Callable] = None,
                     **kwargs) -> Tuple[dict, torch.Tensor]:
        from mmedit.datasets.pipelines import Compose

        if isinstance(imgs, (list, tuple)):
            if not isinstance(imgs[0], (np.ndarray, str)):
                raise AssertionError('imgs must be strings or numpy arrays')
        elif isinstance(imgs, (np.ndarray, str)):
            imgs = [imgs]
        else:
            raise AssertionError('imgs must be strings or numpy arrays')

        cfg = process_model_config(self.model_cfg, imgs, input_shape)

        test_pipeline = Compose(cfg.test_pipeline)

        data_arr = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                data = dict(gt_img=img)
            else:
                data = dict(gt_img_path=img)

            data = test_pipeline(data)
            data_arr.append(data)

        data = collate(data_arr, samples_per_gpu=len(imgs))

        if self.device != 'cpu':
            data = scatter(data, [self.device])[0]

        return data, (data['masked_img'], data['mask'])

    def visualize(self,
                  model: nn.Module,
                  image: Union[str, np.ndarray],
                  result: Union[list, np.ndarray],
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  **kwargs) -> np.ndarray:
        if len(result.shape) == 4:
            result = result[0]

        with torch.no_grad():
            result = result.transpose(1, 2, 0)
            result = np.clip(result, 0, 1)[:, :, ::-1]
            result = (result * 255.0).round()

            output_file = None if show_result else output_file

            if show_result:
                int_result = result.astype(np.uint8)
                mmcv.imshow(int_result, window_name, 0)
            if output_file is not None:
                mmcv.imwrite(result, output_file)

        if not (show_result or output_file):
            warnings.warn(
                'show_result==False and output_file is not specified, only '
                'result image will be returned')
            return result

    @staticmethod
    def run_inference(model: nn.Module, model_inputs: dict) -> list:
        results = model(model_inputs['masked_img'], model_inputs['mask'])
        if not isinstance(results[0], np.ndarray):
            results = [results[0].detach().cpu().numpy()]
        return results

    @staticmethod
    def get_partition_cfg(partition_type: str, **kwargs) -> dict:
        raise NotImplementedError

    @staticmethod
    def get_tensor_from_input(input_data: dict) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def evaluate_outputs(model_cfg,
                         outputs: list,
                         dataset: Dataset,
                         metrics: Optional[str] = None,
                         out: Optional[str] = None,
                         metric_options: Optional[dict] = None,
                         format_only: bool = False,
                         log_file: Optional[str] = None,
                         json_file: Optional[str] = None,
                         **kwargs) -> None:
        """Evaluation function implemented in mmedit.

        Args:
            model_cfg (Config): The model config.
            outputs (list): A list of result of model inference.
            dataset (Dataset): Input dataset to run test.
            metrics (str): Evaluation metrics, which depends on
                the codebase and the dataset, e.g., "PSNR", "SSIM" in mmedit.
            out (str): Output result file in pickle format, defaults to `None`.
            metric_options (dict): Custom options for evaluation, will be
                kwargs for dataset.evaluate() function. Defaults to `None`.
            format_only (bool): Format the output results without perform
                evaluation. It is useful when you want to format the result
                to a specific format and submit it to the test server. Defaults
                to `False`.
            log_file (str | None): The file to write the evaluation results.
                Defaults to `None` and the results will only print on stdout.
        """
        from mmcv.utils import get_logger

        logger = get_logger('test', log_file=log_file)

        if out:
            logger.debug(f'writing results to {out}')
            mmcv.dump(outputs, out)

        print('')
        stats = dataset.evaluate(outputs)
        for stat in stats:
            logger.info('Eval-{}: {}'.format(stat, stats[stat]))
        if json_file is not None:
            mmcv.dump(stats, json_file, indent=4)

    def get_preprocess(self) -> list:
        """Get the preprocess information for SDK."""
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        preprocess = model_cfg.test_pipeline
        return preprocess

    def get_postprocess(self) -> dict:
        """Get the postprocess information for SDK."""
        return dict()

    def get_model_name(self) -> str:
        """Get the model name."""
        assert 'generator' in self.model_cfg.model, \
            'generator not in model config'
        assert 'type' in self.model_cfg.model.generator, \
            'generator contains no type'
        name = self.model_cfg.model.generator.type.lower()
        return name
