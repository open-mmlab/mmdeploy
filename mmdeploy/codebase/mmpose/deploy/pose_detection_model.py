# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import Backend, get_backend, load_config


class End2EndModel(BaseBackendModel):
    """End to end model for inference of pose detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        deploy_cfg (str | mmcv.Config): Deployment config file or loaded Config
            object.
        deploy_cfg (str | mmcv.Config): Model config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, mmcv.Config] = None,
                 model_cfg: Union[str, mmcv.Config] = None):
        super(End2EndModel, self).__init__(deploy_cfg=deploy_cfg)
        self.deploy_cfg = deploy_cfg
        self.model_cfg = model_cfg
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend, backend_files, device):
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            output_names=output_names)

    def forward(self, img: torch.Tensor, img_metas: Sequence[Sequence[dict]],
                *args, **kwargs):
        """Run forward inference.

        Args:
            img (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[Sequence[dict]]): A list of meta info for
                image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        input_img = img.contiguous()
        outputs = self.forward_test(input_img, img_metas, *args, **kwargs)
        heatmaps = outputs[0]
        keypoints = self.keypoint_decode(img_metas, heatmaps)
        return keypoints

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs) -> \
            List[np.ndarray]:
        """The interface for forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            List[np.ndarray]: A list of segmentation map.
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = self.wrapper.output_to_list(outputs)
        outputs = [out.detach().cpu().numpy() for out in outputs]
        return outputs

    def keypoint_decode(self, img_metas, heatmaps):
        from mmpose.core.evaluation.top_down_eval import \
            keypoints_from_heatmaps
        batch_size = len(img_metas)
        assert batch_size == 1
        test_cfg = self.model_cfg.model.test_cfg
        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_heatmaps(
            heatmaps,
            c,
            s,
            unbiased=test_cfg.get('unbiased_decoding', False),
            post_process=test_cfg.get('post_process', 'default'),
            kernel=test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=test_cfg.get('valid_radius_factor', 0.0546875),
            use_udp=test_cfg.get('use_udp', False),
            target_type=test_cfg.get('target_type', 'GaussianHeatmap'))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def show_result(self,
                    img: np.ndarray,
                    result: list,
                    win_name: str = '',
                    skeleton: Optional[Sequence[Sequence[int]]] = None,
                    pose_kpt_color: Optional[Sequence[Sequence[int]]] = None,
                    pose_link_color: Optional[Sequence[Sequence[int]]] = None,
                    bbox_color: str = 'green',
                    text_color: str = 'white',
                    radius: int = 4,
                    thickness: int = 1,
                    font_scale: float = 0.5,
                    bbox_thickness: int = 1,
                    show: bool = False,
                    show_keypoint_weight: bool = False,
                    wait_time: int = 0,
                    out_file: Optional[str] = None,
                    **kwargs):
        """Show predictions of pose.

        Args:
            img: (np.ndarray): Input image to draw predictions.
            result (list): A list of predictions.
            win_name (str): The name of visualization window. Default is ''.
            show (bool): Whether to show plotted image in windows. Defaults to
                `True`.
            out_file (str): Output image file to save drawn predictions.

        Returns:
            np.ndarray: Drawn image, only if not `show` or `out_file`.
        """
        from mmpose.core import imshow_bboxes, imshow_keypoints
        from mmcv.image import imwrite
        from mmcv.visualization.image import imshow
        kpt_score_thr = 0.3
        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img


def build_pose_detection_model(model_files: Sequence[str],
                               model_cfg: Union[str, mmcv.Config],
                               deploy_cfg: Union[str, mmcv.Config],
                               device: str, **kwargs):
    """Build object segmentation model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Pose model for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    backend_pose_model = End2EndModel(
        backend,
        model_files,
        device,
        deploy_cfg=deploy_cfg,
        model_cfg=model_cfg,
        **kwargs)

    return backend_pose_model
