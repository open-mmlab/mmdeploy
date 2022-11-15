# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from torch.utils.data import Dataset

from mmdeploy.codebase.base import BaseTask
from mmdeploy.codebase.mmedit.deploy.mmediting import MMEDIT_TASK
from mmdeploy.utils import Task, Backend, get_input_shape, load_config


def process_model_config(model_cfg: mmcv.Config,
                         imgs: Union[Sequence[str], Sequence[np.ndarray]],
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (mmcv.Config): The model config.
        imgs (Sequence[str] | Sequence[np.ndarray]): Input image(s), accepted
            data type are List[str], List[np.ndarray].
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        mmcv.Config: the model config after processing.
    """
    config = load_config(model_cfg)[0].copy()
    keys_to_remove = ['gt', 'gt_path']
    # MMEdit doesn't support LoadImageFromWebcam.
    # Remove "LoadImageFromFile" and related metakeys.
    load_from_file = isinstance(imgs[0], str)
    is_static_cfg = input_shape is not None
    if not load_from_file:
        config.test_pipeline.pop(0)
        keys_to_remove.append('lq_path')

    # Fix the input shape by 'Resize'
    if is_static_cfg:
        resize = {
            'type': 'Resize',
            'scale': (input_shape[0], input_shape[1]),
            'keys': ['lq']
        }
        config.test_pipeline.insert(1, resize)

    for key in keys_to_remove:
        for pipeline in list(config.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                config.test_pipeline.remove(pipeline)
            if 'keys' in pipeline:
                while key in pipeline['keys']:
                    pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    config.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline:
                while key in pipeline['meta_keys']:
                    pipeline['meta_keys'].remove(key)
    return config


@MMEDIT_TASK.register_module(Task.SUPER_RESOLUTION.value)
class SuperResolution(BaseTask):
    """BaseTask class of super resolution task.

    Args:
        model_cfg (mmcv.Config): Model config file.
        deploy_cfg (mmcv.Config): Deployment config file.
        device (str): A string specifying device type.
    """

    def __init__(self, model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                 device: str):
        super().__init__(model_cfg, deploy_cfg, device)

    def init_backend_model(self,
                           model_files: Sequence[str] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files. Default is None.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .super_resolution_model import build_super_resolution_model
        model = build_super_resolution_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            **kwargs)
        return model

    def init_pytorch_model(self,
                           model_checkpoint: Optional[str] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize torch model.

        Args:
            model_checkpoint (str): The checkpoint file of torch model,
                defaults to `None`.

        Returns:
            nn.Module: An initialized torch model generated by other OpenMMLab
                codebases.
        """
        from mmedit.apis import init_model
        model = init_model(self.model_cfg, model_checkpoint, self.device)
        model.forward = model.forward_dummy
        return model.eval()

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Optional[Sequence[int]] = None,
                     backend: Optional[Backend] = None,
                     **kwargs) -> Tuple[Dict, torch.Tensor]:
        """Create input for editing processor.

        Args:
            imgs (str | np.ndarray): Input image(s).
            input_shape (Sequence[int] | None): Input shape of image in
                (width, height) format, defaults to `None`.
            backend (Backend | None): Target backend. Default to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
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
                data = dict(lq=img)
            else:
                data = dict(lq_path=img)

            data = test_pipeline(data)
            data_arr.append(data)

        data = collate(data_arr, samples_per_gpu=len(imgs))

        if self.device != 'cpu':
            data = scatter(data, [self.device])[0]

        data['img'] = data['lq']

        return data, data['img']

    def visualize(self,
                  model: torch.nn.Module,
                  image: Union[str, np.ndarray],
                  result: Union[list, np.ndarray],
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  **kwargs) -> np.ndarray:
        """Visualize result of a model.

        Args:
            model (nn.Module): Input model.
            image (str | np.ndarray): Input image to draw predictions on.
            result (list | np.ndarray): A list of result.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows, defaults
                to `False`.
        """
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
    def run_inference(model: torch.nn.Module,
                      model_inputs: Dict[str, torch.Tensor]) -> list:
        """Run inference once for a super resolution model of mmedit.

        Args:
            model (nn.Module): Input model.
            model_inputs (dict): A dict containing model inputs tensor and
                meta info.

        Returns:
            list: The predictions of model inference.
        """
        result = model(model_inputs['lq'])
        if not isinstance(result[0], np.ndarray):
            result = [result[0].detach().cpu().numpy()]
        return result

    @staticmethod
    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        """Get a certain partition config for mmedit.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        raise NotImplementedError

    @staticmethod
    def get_tensor_from_input(input_data: Dict[str, Any]) -> torch.Tensor:
        """Get input tensor from input data.

        Args:
            input_data (dict): Input data containing meta info
            and image tensor.
        Returns:
            torch.Tensor: An image in `Tensor`.
        """
        return input_data['lq']

    @staticmethod
    def evaluate_outputs(model_cfg,
                         outputs: list,
                         dataset: Dataset,
                         metrics: Optional[str] = None,
                         out: Optional[str] = None,
                         metric_options: Optional[dict] = None,
                         format_only: bool = False,
                         log_file: Optional[str] = None,
                         **kwargs) -> None:
        """Evaluation function implemented in mmedit.

        Args:
            model_cfg (mmcv.Config): The model config.
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
        # The Dataset doesn't need metrics
        # print metrics
        stats = dataset.evaluate(outputs)
        for stat in stats:
            logger.info('Eval-{}: {}'.format(stat, stats[stat]))

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        preprocess = model_cfg.test_pipeline
        for item in preprocess:
            if 'Normalize' == item['type'] and 'std' in item:
                item['std'] = [255, 255, 255]
        return preprocess

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Nonthing for super resolution.
        """
        return dict()

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'generator' in self.model_cfg.model, 'generator not in model '
        'config'
        assert 'type' in self.model_cfg.model.generator, 'generator contains '
        'no type'
        name = self.model_cfg.model.generator.type.lower()
        return name
