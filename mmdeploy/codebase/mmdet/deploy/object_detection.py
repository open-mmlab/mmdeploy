# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer
from torch.utils.data import Dataset

from mmdeploy.utils import Task
from mmdeploy.utils.config_utils import get_input_shape, is_dynamic_shape
from ...base import BaseTask
from .mmdetection import MMDET_TASK


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
    from mmdet.datasets import replace_ImageToTensor

    cfg = model_cfg.copy()

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    # for static exporting
    if input_shape is not None:
        cfg.data.test.pipeline[1]['img_scale'] = tuple(input_shape)
        transforms = cfg.data.test.pipeline[1]['transforms']
        for trans in transforms:
            trans_type = trans['type']
            if trans_type == 'Resize':
                trans['keep_ratio'] = False
            elif trans_type == 'Pad':
                trans['size_divisor'] = 1

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    return cfg


@MMDET_TASK.register_module(Task.OBJECT_DETECTION.value)
class ObjectDetection(BaseTask):

    def __init__(self, model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                 device: str) -> None:
        super().__init__(model_cfg, deploy_cfg, device)

    def init_backend_model(self,
                           model_files: Optional[str] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .object_detection_model import build_object_detection_model
        model = build_object_detection_model(
            model_files, self.model_cfg, self.deploy_cfg, device=self.device)
        return model.eval()

    def init_pytorch_model(self,
                           model_checkpoint: Optional[str] = None,
                           cfg_options: Optional[Dict] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize torch model.

        Args:
            model_checkpoint (str): The checkpoint file of torch model,
                defaults to `None`.
            cfg_options (dict): Optional config key-pair parameters.

        Returns:
            nn.Module: An initialized torch model generated by other OpenMMLab
                codebases.
        """
        if self.from_mmrazor:
            from mmrazor.apis import init_mmdet_model as init_detector
        else:
            from mmdet.apis import init_detector

        model = init_detector(self.model_cfg, model_checkpoint, self.device,
                              cfg_options)
        return model.eval()

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Sequence[int] = None) \
            -> Tuple[Dict, torch.Tensor]:
        """Create input for detector.

        Args:
            imgs (str|np.ndarray): Input image(s), accpeted data type are
                `str`, `np.ndarray`.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Defaults to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        from mmcv.parallel import collate, scatter
        from mmdet.datasets.pipelines import Compose
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        dynamic_flag = is_dynamic_shape(self.deploy_cfg)
        cfg = process_model_config(self.model_cfg, imgs, input_shape)
        # Drop pad_to_square when static shape. Because static shape should
        # ensure the shape before input image.
        if not dynamic_flag:
            transform = cfg.data.test.pipeline[1]
            if 'transforms' in transform:
                transform_list = transform['transforms']
                for i, step in enumerate(transform_list):
                    if step['type'] == 'Pad' and 'pad_to_square' in step \
                       and step['pad_to_square']:
                        transform_list.pop(i)
                        break
        test_pipeline = Compose(cfg.data.test.pipeline)
        data_list = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = test_pipeline(data)
            data_list.append(data)

        data = collate(data_list, samples_per_gpu=len(imgs))

        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]
        if self.device != 'cpu':
            data = scatter(data, [self.device])[0]

        return data, data['img']

    def visualize(self,
                  model: torch.nn.Module,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str,
                  show_result: bool = False,
                  score_thr: float = 0.3):
        """Visualize predictions of a model.

        Args:
            model (nn.Module): Input model.
            image (str | np.ndarray): Input image to draw predictions on.
            result (list): A list of predictions.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows, defaults
                to `False`.
            score_thr (float): The score threshold to display the bbox.
                Defaults to 0.3.
        """
        show_img = mmcv.imread(image) if isinstance(image, str) else image
        output_file = None if show_result else output_file
        model.show_result(
            show_img,
            result=result,
            win_name=window_name,
            show=show_result,
            out_file=output_file,
            score_thr=score_thr)

    @staticmethod
    def run_inference(model: torch.nn.Module,
                      model_inputs: Dict[str, torch.Tensor]) -> list:
        """Run inference once for a object detection model of mmdet.

        Args:
            model (nn.Module): Input model.
            model_inputs (dict): A dict containing model inputs tensor and
                meta info.

        Returns:
            list: The predictions of model inference.
        """
        return model(**model_inputs, return_loss=False, rescale=True)

    @staticmethod
    def get_partition_cfg(partition_type: str) -> Dict:
        """Get a certain partition config for mmdet.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        from .model_partition_cfg import MMDET_PARTITION_CFG
        assert (partition_type in MMDET_PARTITION_CFG), \
            f'Unknown partition_type {partition_type}'
        return MMDET_PARTITION_CFG[partition_type]

    @staticmethod
    def get_tensor_from_input(input_data: Dict[str, Any]) -> torch.Tensor:
        """Get input tensor from input data.

        Args:
            input_data (dict): Input data containing meta info and image
                tensor.
        Returns:
            torch.Tensor: An image in `Tensor`.
        """
        img_tensor = input_data['img'][0]
        if isinstance(img_tensor, DataContainer):
            img_tensor = img_tensor.data[0]
        return img_tensor

    @staticmethod
    def evaluate_outputs(model_cfg: mmcv.Config,
                         outputs: Sequence,
                         dataset: Dataset,
                         metrics: Optional[str] = None,
                         out: Optional[str] = None,
                         metric_options: Optional[dict] = None,
                         format_only: bool = False,
                         log_file: Optional[str] = None):
        """Perform post-processing to predictions of model.

        Args:
            model_cfg (mmcv.Config): Model config.
            outputs (list): A list of predictions of model inference.
            dataset (Dataset): Input dataset to run test.
            metrics (str): Evaluation metrics, which depends on
                the codebase and the dataset, e.g., "bbox", "segm", "proposal"
                for COCO, and "mAP", "recall" for PASCAL VOC in mmdet.
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
        from mmdeploy.utils.logging import get_logger
        logger = get_logger('test', log_file=log_file)

        if out:
            logger.debug(f'writing results to {out}')
            mmcv.dump(outputs, out)
        kwargs = {} if metric_options is None else metric_options
        if format_only:
            dataset.format_results(outputs, **kwargs)
        if metrics:
            eval_kwargs = model_cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=metrics, **kwargs))
            logger.info(dataset.evaluate(outputs, **eval_kwargs))

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        preprocess = model_cfg.data.test.pipeline
        return preprocess

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        postprocess = self.model_cfg.model.test_cfg
        if 'rpn' in postprocess:
            postprocess['min_bbox_size'] = postprocess['rpn']['min_bbox_size']
        if 'rcnn' in postprocess:
            postprocess['score_thr'] = postprocess['rcnn']['score_thr']
            if 'mask_thr_binary' in postprocess['rcnn']:
                postprocess['mask_thr_binary'] = postprocess['rcnn'][
                    'mask_thr_binary']
        return postprocess

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name
