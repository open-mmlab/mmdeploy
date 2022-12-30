# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer, collate, scatter
from torch import nn
from torch.utils.data import Dataset

from mmdeploy.codebase.base import BaseTask
from mmdeploy.utils import Task, get_input_shape
from .mmrotate import MMROTATE_TASK


def replace_RResize(pipelines):
    """Rename RResize to Resize.

    args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all RResize renamed to
            Resize.
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_RResize(pipeline['transforms'])
        elif pipeline.type == 'RResize':
            pipelines[i].type = 'Resize'
            if 'keep_ratio' not in pipelines[i]:
                pipelines[i]['keep_ratio'] = True  # default value
    return pipelines


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

    cfg = copy.deepcopy(model_cfg)

    if isinstance(imgs[0], np.ndarray):
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    # for static exporting
    if input_shape is not None:
        cfg.data.test.pipeline[1]['img_scale'] = tuple(input_shape)
        transforms = cfg.data.test.pipeline[1]['transforms']
        for trans in transforms:
            trans_type = trans['type']
            if trans_type == 'Pad' and 'size_divisor' in trans:
                trans['size_divisor'] = 1

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    return cfg


@MMROTATE_TASK.register_module(Task.ROTATED_DETECTION.value)
class RotatedDetection(BaseTask):
    """Rotated detection task class.

    Args:
        model_cfg (mmcv.Config): Loaded model Config object..
        deploy_cfg (mmcv.Config): Loaded deployment Config object.
        device (str): A string represents device type.
    """

    def __init__(self, model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                 device: str):
        super(RotatedDetection, self).__init__(model_cfg, deploy_cfg, device)

    def init_backend_model(self,
                           model_files: Optional[str] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .rotated_detection_model import build_rotated_detection_model
        model = build_rotated_detection_model(
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
            nn.Module: An initialized torch model generated by OpenMMLab
                codebases.
        """
        import warnings

        from mmcv.runner import load_checkpoint
        from mmdet.core import get_classes
        from mmrotate.models import build_detector

        if isinstance(self.model_cfg, str):
            self.model_cfg = mmcv.Config.fromfile(self.model_cfg)
        elif not isinstance(self.model_cfg, mmcv.Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(self.model_cfg)}')
        if cfg_options is not None:
            self.model_cfg.merge_from_dict(cfg_options)
        self.model_cfg.model.pretrained = None
        self.model_cfg.model.train_cfg = None
        model = build_detector(
            self.model_cfg.model, test_cfg=self.model_cfg.get('test_cfg'))
        if model_checkpoint is not None:
            map_loc = 'cpu' if self.device == 'cpu' else None
            checkpoint = load_checkpoint(
                model, model_checkpoint, map_location=map_loc)
            if 'CLASSES' in checkpoint.get('meta', {}):
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                warnings.simplefilter('once')
                warnings.warn('Class names are not saved in the checkpoint\'s '
                              'meta data, use COCO classes by default.')
                model.CLASSES = get_classes('coco')
        model.cfg = self.model_cfg
        model.to(self.device)
        return model.eval()

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Optional[Sequence[int]] = None,
                     pipeline_updater: Optional[Callable] = None, **kwargs) \
            -> Tuple[Dict, torch.Tensor]:
        """Create input for rotated object detection.

        Args:
            imgs (str | np.ndarray): Input image(s), accepted data type are
            `str`, `np.ndarray`.
            input_shape (Sequence[int] | None): Input shape of image in
                (width, height) format, defaults to `None`.
            pipeline_updater (function | None): A function to get a new
                pipeline.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        from mmdet.datasets.pipelines import Compose

        if isinstance(imgs, (list, tuple)):
            if not isinstance(imgs[0], (np.ndarray, str)):
                raise AssertionError('imgs must be strings or numpy arrays')

        elif isinstance(imgs, (np.ndarray, str)):
            imgs = [imgs]
        else:
            raise AssertionError('imgs must be strings or numpy arrays')
        cfg = process_model_config(self.model_cfg, imgs, input_shape)
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
            # get tensor from list to stack for batch mode (rotated detection)
            data_list.append(data)

        batch_data = collate(data_list, samples_per_gpu=len(imgs))

        for k, v in batch_data.items():
            # batch_size > 1
            if isinstance(v[0], DataContainer):
                batch_data[k] = v[0].data

        if self.device != 'cpu':
            batch_data = scatter(batch_data, [self.device])[0]

        return batch_data, batch_data['img']

    def visualize(self,
                  model: nn.Module,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False):
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
        """
        show_img = mmcv.imread(image) if isinstance(image, str) else image
        output_file = None if show_result else output_file
        model.show_result(
            show_img,
            result,
            out_file=output_file,
            win_name=window_name,
            show=show_result)

    @staticmethod
    def run_inference(model: nn.Module,
                      model_inputs: Dict[str, torch.Tensor]) -> list:
        """Run inference once for a segmentation model of mmseg.

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
        """Get a certain partition config.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        raise NotImplementedError('Not supported yet.')

    @staticmethod
    def get_tensor_from_input(input_data: Dict[str, Any]) -> torch.Tensor:
        """Get input tensor from input data.

        Args:
            input_data (dict): Input data containing meta info and image
                tensor.
        Returns:
            torch.Tensor: An image in `Tensor`.
        """
        img_data = input_data['img'][0]
        if isinstance(img_data, DataContainer):
            return img_data.data[0]
        return img_data

    @staticmethod
    def evaluate_outputs(model_cfg,
                         outputs: Sequence,
                         dataset: Dataset,
                         metrics: Optional[str] = None,
                         out: Optional[str] = None,
                         metric_options: Optional[dict] = None,
                         format_only: bool = False,
                         log_file: Optional[str] = None,
                         json_file: Optional[str] = None):
        """Perform post-processing to predictions of model.

        Args:
            outputs (Sequence): A list of predictions of model inference.
            dataset (Dataset): Input dataset to run test.
            model_cfg (mmcv.Config): The model config.
            metrics (str): Evaluation metrics, which depends on
                the codebase and the dataset, e.g.,  "mAP" for rotated
                detection.
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
        kwargs = {} if metric_options is None else metric_options
        if format_only:
            dataset.format_results(outputs, **kwargs)
        if metrics:
            eval_kwargs = model_cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=metrics, **kwargs))
            results = dataset.evaluate(outputs, **eval_kwargs)
            if json_file is not None:
                mmcv.dump(results, json_file, indent=4)
            logger.info(results)

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        # rename sdk RResize -> Resize
        model_cfg.data.test.pipeline = replace_RResize(
            model_cfg.data.test.pipeline)
        preprocess = model_cfg.data.test.pipeline
        return preprocess

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        postprocess = self.model_cfg.model.test_cfg
        return postprocess

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name

    def update_deploy_config(self,
                             deploy_config: Any,
                             model_type: str,
                             is_dynamic_batch: bool = False,
                             is_dynamic_size: bool = False,
                             input_shape: Optional[Tuple[int]] = None,
                             detection_mode: str = 'detection',
                             *args,
                             **kwargs):

        from mmdeploy.backend.base import get_backend_manager
        from mmdeploy.utils import Backend, get_backend, get_ir_config
        assert detection_mode in ['detection', 'instance-segmentation']

        def _round_up(val, divisor):
            return int(np.ceil(val / divisor) * divisor)

        def _shape_inference():
            nonlocal is_dynamic_size
            nonlocal input_shape
            pipeline = self.model_cfg.data.test.pipeline
            pipeline = pipeline.copy()

            img_scale = None
            if pipeline[1]['type'] == 'MultiScaleFlipAug':
                transforms = pipeline[1]['transforms']
                img_scale = pipeline[1]['img_scale']
            else:
                transforms = pipeline

            resize_trans = None
            pad_trans = None
            for trans in transforms:
                if trans['type'] == 'RResize':
                    resize_trans = trans
                if trans['type'] == 'Pad':
                    pad_trans = trans

            if resize_trans is not None and 'img_scale' in resize_trans:
                img_scale = resize_trans['img_scale']

            if img_scale is None:
                assert input_shape is not None
                img_scale = input_shape

            min_size = img_scale
            opt_size = img_scale
            max_size = img_scale

            if resize_trans is not None:
                min_s = min(img_scale)
                max_s = max(img_scale)
                min_size = tuple((min_s // 4, min_s // 4))
                max_size = tuple((max_s, max_s))

            if pad_trans is not None:
                size = pad_trans.get('size', None)
                size_divisor = pad_trans.get('size_divisor', None)
                pad_to_square = pad_trans.get('pad_to_square', False)

                if size is not None:
                    min_size = size
                    opt_size = size
                    max_size = size
                elif size_divisor is not None:
                    min_size = tuple(
                        _round_up(s, size_divisor) for s in min_size)
                    opt_size = tuple(
                        _round_up(s, size_divisor) for s in opt_size)
                    max_size = tuple(
                        _round_up(s, size_divisor) for s in max_size)
                elif pad_to_square:
                    min_s = min(img_scale)
                    max_s = max(img_scale)
                    min_size = (min_s, min_s)
                    opt_size = (max_s, max_s)
                    max_size = (max_s, max_s)

            if min_size[0] == opt_size[0] == max_size[0] and min_size[
                    1] == opt_size[1] == max_size[1]:
                is_dynamic_size = False
                input_shape = opt_size

            if not is_dynamic_size and input_shape is None:
                input_shape = opt_size

            return min_size, opt_size, max_size

        def _get_mean_std():
            pipeline = self.model_cfg.data.test.pipeline
            pipeline = pipeline.copy()

            if pipeline[1]['type'] == 'MultiScaleFlipAug':
                transforms = pipeline[1]['transforms']
            else:
                transforms = pipeline

            for trans in transforms:
                if trans['type'] == 'Normalize':
                    mean = trans.get('mean', [0.0, 0.0, 0.0])
                    std = trans.get('std', [1.0, 1.0, 1.0])
                    to_rgb = trans.get('to_rgb', False)
                    if to_rgb:
                        mean = mean[::-1]
                        std = std[::-1]
                    return mean, std
            return None, None

        min_size, opt_size, max_size = _shape_inference()
        mean, std = _get_mean_std()

        # update codebase_config
        codebase_config = deploy_config.codebase_config
        codebase_config['update_config'] = False
        post_processing = dict(
            score_threshold=0.05,
            confidence_threshold=0.005,
            iou_threshold=0.5,
            max_output_boxes_per_class=200,
            pre_top_k=5000,
            keep_top_k=100,
            background_label_id=-1)
        codebase_config['model_type'] = model_type
        codebase_config['is_dynamic_batch'] = is_dynamic_batch
        codebase_config['is_dynamic_size'] = is_dynamic_size
        codebase_config['input_shape'] = input_shape
        codebase_config['detection_mode'] = detection_mode
        if 'post_processing' not in codebase_config:
            codebase_config['post_processing'] = post_processing
        deploy_config['codebase_config'] = codebase_config

        # update ir_config
        ir_config = get_ir_config(deploy_config)
        input_names = ['input']
        output_names = ['dets', 'labels']
        if detection_mode == 'instance-segmentation':
            output_names += ['masks']

        ir_config['input_names'] = input_names
        ir_config['output_names'] = output_names

        if is_dynamic_batch or is_dynamic_size:
            dynamic_axes = dict()
            input_axes = dict()
            dets_axes = dict()
            labels_axes = dict()
            if is_dynamic_batch:
                input_axes[0] = 'batch'
                dets_axes[0] = 'batch'
                labels_axes[0] = 'batch'
            if is_dynamic_size:
                input_axes[2] = 'height'
                input_axes[3] = 'width'
                dets_axes[1] = 'num_dets'
                labels_axes[1] = 'num_dets'
            dynamic_axes['input'] = input_axes
            dynamic_axes['dets'] = dets_axes
            dynamic_axes['labels'] = labels_axes
            if detection_mode == 'instance-segmentation':
                masks_axes = dict()
                if is_dynamic_batch:
                    masks_axes[0] = 'batch'
                if is_dynamic_size:
                    masks_axes[1] = 'num_dets'
                dynamic_axes['masks'] = masks_axes
            ir_config['dynamic_axes'] = dynamic_axes
        if input_shape is not None:
            ir_config['input_shape'] = input_shape
        deploy_config['ir_config'] = ir_config

        # update backend_config
        backend = get_backend(deploy_config)
        backend_mgr = get_backend_manager(backend.value)

        min_batch = 1
        opt_batch = 1
        max_batch = 1
        if is_dynamic_batch:
            max_batch = 2
        num_channel = 3

        if not is_dynamic_size:
            assert input_shape is not None
            min_shape = (min_batch, num_channel, *input_shape[::-1])
            opt_shape = (opt_batch, num_channel, *input_shape[::-1])
            max_shape = (max_batch, num_channel, *input_shape[::-1])
        else:
            min_shape = (min_batch, num_channel, *min_size[::-1])
            opt_shape = (opt_batch, num_channel, *opt_size[::-1])
            max_shape = (max_batch, num_channel, *max_size[::-1])

        if backend == Backend.TENSORRT:
            backend_mgr.update_deploy_config(
                deploy_config,
                opt_shapes=dict(input=opt_shape),
                min_shapes=dict(input=min_shape),
                max_shapes=dict(input=max_shape))
        elif backend == Backend.SDK:
            backend_mgr.update_deploy_config(
                deploy_config,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=['filename', 'ori_shape'])
                ])
        else:
            backend_mgr.update_deploy_config(
                deploy_config,
                opt_shapes=dict(input=opt_shape),
                dtypes=dict(input='float32'),
                input_names=input_names,
                mean=mean,
                std=std)

        return deploy_config
