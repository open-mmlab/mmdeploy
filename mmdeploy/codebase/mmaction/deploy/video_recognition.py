# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer
from torch.utils.data import Dataset

from mmdeploy.codebase.base import BaseTask
from mmdeploy.utils import (Task, get_backend_config, get_input_shape,
                            get_root_logger)
from mmdeploy.utils.dataset import is_can_sort_dataset, sort_dataset
from .mmaction import MMACTION_TASK


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
    cfg = model_cfg.deepcopy()
    test_pipeline = cfg.data.test.pipeline

    if 'Init' not in test_pipeline[0]['type']:
        test_pipeline = [dict(type='OpenCVInit')] + test_pipeline
    else:
        test_pipeline[0] = dict(type='OpenCVInit')
    for i in range(len(test_pipeline)):
        if 'Decode' in test_pipeline[i]['type']:
            test_pipeline[i] = dict(type='OpenCVDecode')
    cfg.data.test.pipeline = test_pipeline
    return cfg


@MMACTION_TASK.register_module(Task.VIDEO_RECOGNITION.value)
class VideoRecognition(BaseTask):
    """Video recognition task class.

    Args:
        model_cfg (mmcv.Config): Original PyTorch model config file.
        deploy_cfg (mmcv.Config): Deployment config file or loaded Config
            object.
        device (str): A string represents device type.
    """

    def init_backend_model(self,
                           model_files: Optional[str] = None,
                           **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .video_recognition_model import build_video_recognition_model
        model = build_video_recognition_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            **kwargs)
        return model.eval()

    def build_dataset(self,
                      dataset_cfg: Union[str, mmcv.Config],
                      dataset_type: str = 'val',
                      is_sort_dataset: bool = False,
                      **kwargs) -> Dataset:
        """Build dataset for different codebase.

        Args:
            dataset_cfg (str | mmcv.Config): Dataset config file or Config
                object.
            dataset_type (str): Specifying dataset type, e.g.: 'train', 'test',
                'val', defaults to 'val'.
            is_sort_dataset (bool): When 'True', the dataset will be sorted
                by image shape in ascending order if 'dataset_cfg'
                contains information about height and width.
                Default is `False`.

        Returns:
            Dataset: The built dataset.
        """

        backend_cfg = get_backend_config(self.deploy_cfg)
        if 'pipeline' in backend_cfg:
            ori = dataset_cfg.data[dataset_type].pipeline
            be = backend_cfg.pipeline
            index_ori = -1
            index_be = -1
            for i, trans in enumerate(ori):
                if trans['type'] == 'SampleFrames':
                    index_ori = i
                    break
            for i, trans in enumerate(be):
                if trans['type'] == 'SampleFrames':
                    index_be = i
                    break
            if index_ori != -1 and index_be != -1:
                be[index_be] = ori[index_ori]

            dataset_cfg.data[dataset_type].pipeline = be
        dataset = self.codebase_class.build_dataset(dataset_cfg, dataset_type,
                                                    **kwargs)
        logger = get_root_logger()
        if is_sort_dataset:
            if is_can_sort_dataset(dataset):
                sort_dataset(dataset)
            else:
                logger.info('Sorting the dataset by \'height\' and \'width\' '
                            'is not possible.')
        return dataset

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
        from mmaction.apis import init_recognizer
        from mmcv.cnn.utils import revert_sync_batchnorm
        model = init_recognizer(self.model_cfg, model_checkpoint, self.device)
        model = revert_sync_batchnorm(model)
        return model.eval()

    def create_input(self,
                     imgs: Union[str, np.ndarray, Sequence],
                     input_shape: Optional[Sequence[int]] = None,
                     pipeline_updater: Optional[Callable] = None,
                     **kwargs) -> Tuple[Dict, torch.Tensor]:
        """Create input for recognizer.

        Args:
            imgs (Any): Input image(s), accepted data type are `str`,
                `np.ndarray`, `torch.Tensor`.
            input_shape (Sequence[int] | None): Input shape of image in
                (width, height) format, defaults to `None`.
            pipeline_updater (function | None): A function to get a new
                pipeline.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """

        if isinstance(imgs, (list, tuple)):
            if not all(isinstance(img, str) for img in imgs):
                raise AssertionError('imgs must be strings')
        elif isinstance(imgs, str):
            imgs = [imgs]
        else:
            raise AssertionError('imgs must be strings')

        from mmaction.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        if isinstance(imgs, (str, np.ndarray)):
            imgs = [imgs]
        cfg = process_model_config(self.model_cfg, imgs, input_shape)
        test_pipeline = Compose(cfg.data.test.pipeline)

        data_list = []
        for img in imgs:
            # prepare data
            data = dict(filename=img, label=-1, start_index=0, modality='RGB')
            # build the data pipeline
            data = test_pipeline(data)
            data_list.append(data)

        batch_data = collate(data_list, samples_per_gpu=len(imgs))
        if self.device != 'cpu':
            batch_data = scatter(batch_data, [self.device])[0]

        for k, v in batch_data.items():
            # batch_size > 1
            if isinstance(v[0], DataContainer):
                batch_data[k] = v[0].data
        return batch_data, batch_data['imgs']

    def visualize(self,
                  model,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  opacity: float = 0.5):
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
            opacity: (float): Opacity of painted segmentation map.
                    Defaults to `0.5`.
        """
        logger = get_root_logger()
        if not isinstance(image, str):
            logger.warning('Input should be a video path')
            return
        import cv2
        cap = cv2.VideoCapture(image)
        _, img = cap.read()
        top1 = np.argsort(result)[-1]
        text = f'top1-label: {top1}'
        cv2.putText(img, text, (0, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (0, 0, 0), 1, 1)
        cv2.imwrite(output_file, img)

    @staticmethod
    def run_inference(model, model_inputs: Dict[str, torch.Tensor]):
        """Run inference once for a video recognition model of mmaction.

        Args:
            model (nn.Module): Input model.
            model_inputs (dict): A dict containing model inputs tensor and
                meta info.

        Returns:
            list: The predictions of model inference.
        """
        return model(**model_inputs, return_loss=False)

    @staticmethod
    def get_partition_cfg(partition_type: str) -> Dict:
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
        return input_data['imgs'][0]

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
            outputs (list): A list of predictions of model inference.
            dataset (Dataset): Input dataset to run test.
            model_cfg (mmcv.Config): The model config.
            metrics (str): Evaluation metrics, which depends on
                the codebase and the dataset.
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
            results = dataset.evaluate(
                outputs, metrics, logger=logger, **kwargs)
            if json_file is not None:
                mmcv.dump(results, json_file, indent=4)

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        cfg = process_model_config(self.model_cfg, [''], input_shape)
        pipeline = cfg.data.test.pipeline

        lift = dict(type='Lift', transforms=[])
        lift['transforms'].append(dict(type='LoadImageFromFile'))
        transforms2index = {}
        for i, trans in enumerate(pipeline):
            transforms2index[trans['type']] = i
        lift_key = [
            'Resize', 'Normalize', 'TenCrop', 'ThreeCrop', 'CenterCrop'
        ]
        for key in lift_key:
            if key in transforms2index:
                index = transforms2index[key]
                if key == 'Normalize':
                    pipeline[index]['to_rgb'] = True
                if key == 'Resize' and 'scale' in pipeline[index]:
                    value = pipeline[index].pop('scale')
                    if len(value) == 2 and value[0] == -1:
                        value = value[::-1]
                    pipeline[index]['size'] = value
                lift['transforms'].append(pipeline[index])

        other = []
        must_key = ['FormatShape', 'Collect']
        for key in must_key:
            assert key in transforms2index
            index = transforms2index[key]
            if key == 'Collect':
                pipeline[index]['keys'] = ['img']
            other.append(pipeline[index])

        reorder = [lift, *other]
        return reorder

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        assert 'cls_head' in self.model_cfg.model
        assert 'num_classes' in self.model_cfg.model.cls_head
        num_classes = self.model_cfg.model.cls_head.num_classes
        postprocess = dict(topk=1, num_classes=num_classes)
        return postprocess

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'backbone' in self.model_cfg.model, 'backbone not in model ' \
            'config'
        assert 'type' in self.model_cfg.model.backbone, 'backbone contains ' \
            'no type'
        name = self.model_cfg.model.backbone.type.lower()
        return name

    def update_deploy_config(self,
                             deploy_config: Any,
                             model_type: str,
                             is_dynamic_batch: bool = False,
                             *args,
                             **kwargs):
        from mmdeploy.backend.base import get_backend_manager
        from mmdeploy.utils import Backend, get_backend, get_ir_config

        def _get_input_shape(channel=3):
            pipeline = self.model_cfg.data.test.pipeline
            pipeline = pipeline.copy()

            transforms = pipeline

            sample_frame_trans = None
            center_crop_trans = None
            three_crop_trans = None
            ten_crop_trans = None
            format_shape_trans = None

            for trans in transforms:
                if trans['type'] == 'SampleFrames':
                    sample_frame_trans = trans
                if trans['type'] == 'CenterCrop':
                    center_crop_trans = trans
                if trans['type'] == 'ThreeCrop':
                    three_crop_trans = trans
                if trans['type'] == 'TenCrop':
                    ten_crop_trans = trans
                if trans['type'] == 'FormatShape':
                    format_shape_trans = trans

            assert sample_frame_trans is not None
            assert format_shape_trans is not None

            clip_len = sample_frame_trans['clip_len']
            num_clips = sample_frame_trans.get('num_clips', 1)
            input_format = format_shape_trans['input_format']

            if center_crop_trans is not None:
                num_crop = 1
                input_size = center_crop_trans['crop_size']
            elif three_crop_trans is not None:
                num_crop = 3
                input_size = three_crop_trans['crop_size']
            elif ten_crop_trans is not None:
                num_crop = 10
                input_size = ten_crop_trans['crop_size']
            else:
                raise ValueError('Unknown crop.')

            if isinstance(input_size, int):
                input_size = (input_size, input_size)

            if input_format == 'NCHW':
                return [
                    clip_len * num_clips * num_crop, channel, *input_size[::-1]
                ]
            elif input_format == 'NCTHW':
                return [
                    num_clips * num_crop, channel, clip_len, *input_size[::-1]
                ]
            else:
                raise ValueError(f'Unknown input_format: {input_format}.')

        def _get_mean_std():
            pipeline = self.model_cfg.data.test.pipeline
            pipeline = pipeline.copy()
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
            return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

        num_channel = 3
        input_shape = _get_input_shape(num_channel)
        mean, std = _get_mean_std()

        input_names = ['input']
        output_names = ['output']

        ir_config = get_ir_config(deploy_config)
        ir_config['input_names'] = input_names
        ir_config['output_names'] = output_names

        if is_dynamic_batch:
            dynamic_axes = dict()
            input_axes = dict()
            output_axes = dict()

            if is_dynamic_batch:
                input_axes[0] = 'batch'
                output_axes[0] = 'batch'

            dynamic_axes['input'] = input_axes
            dynamic_axes['output'] = output_axes
            ir_config['dynamic_axes'] = dynamic_axes

        deploy_config['ir_config'] = ir_config

        backend = get_backend(deploy_config)
        backend_mgr = get_backend_manager(backend.value)

        min_batch = 1
        opt_batch = 1
        max_batch = 1
        if is_dynamic_batch:
            max_batch = 2

        min_shape = (min_batch, *input_shape)
        opt_shape = (opt_batch, *input_shape)
        max_shape = (max_batch, *input_shape)

        if backend == Backend.SDK:
            deploy_config = backend_mgr.update_deploy_config(
                deploy_config,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=['filename', 'ori_shape'])
                ])
        else:
            deploy_config = backend_mgr.update_deploy_config(
                deploy_config,
                opt_shapes=dict(input=opt_shape),
                min_shapes=dict(input=min_shape),
                max_shapes=dict(input=max_shape),
                dtypes=dict(input='float32'),
                input_names=input_names,
                mean=mean,
                std=std)

        return deploy_config
