from functools import partial
from typing import Union

import mmcv
import numpy as np
import torch
from mmdet.core import bbox2result
from mmdet.datasets import DATASETS
from mmdet.models import BaseDetector

from mmdeploy.mmdet.core.post_processing import multiclass_nms
from mmdeploy.utils.config_utils import Backend, get_backend, load_config


class DeployBaseDetector(BaseDetector):
    """DeployBaseDetector."""

    def __init__(self, class_names, device_id, **kwargs):
        super(DeployBaseDetector, self).__init__()
        self.CLASSES = class_names
        self.device_id = device_id

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def val_step(self, data, optimizer):
        raise NotImplementedError('This method is not implemented.')

    def train_step(self, data, optimizer):
        raise NotImplementedError('This method is not implemented.')

    def aforward_test(self, *, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def forward(self, img, img_metas, *args, **kwargs):
        outputs = self.forward_test(img, img_metas, *args, **kwargs)
        batch_dets, batch_labels = outputs[:2]
        batch_masks = outputs[2] if len(outputs) == 3 else None
        batch_size = img[0].shape[0]
        img_metas = img_metas[0]
        results = []
        rescale = kwargs.get('rescale', True)
        for i in range(batch_size):
            dets, labels = batch_dets[i], batch_labels[i]
            if rescale:
                scale_factor = img_metas[i]['scale_factor']

                if isinstance(scale_factor, (list, tuple, np.ndarray)):
                    assert len(scale_factor) == 4
                    scale_factor = np.array(scale_factor)[None, :]  # [1,4]
                dets[:, :4] /= scale_factor

            if 'border' in img_metas[i]:
                # offset pixel of the top-left corners between original image
                # and padded/enlarged image, 'border' is used when exporting
                # CornerNet and CentripetalNet to onnx
                x_off = img_metas[i]['border'][2]
                y_off = img_metas[i]['border'][0]
                dets[:, [0, 2]] -= x_off
                dets[:, [1, 3]] -= y_off
                dets[:, :4] *= (dets[:, :4] > 0).astype(dets.dtype)

            dets_results = bbox2result(dets, labels, len(self.CLASSES))

            if batch_masks is not None:
                masks = batch_masks[i]
                img_h, img_w = img_metas[i]['img_shape'][:2]
                ori_h, ori_w = img_metas[i]['ori_shape'][:2]
                masks = masks[:, :img_h, :img_w]
                # avoid to resize masks with zero dim
                if rescale and masks.shape[0] != 0:
                    masks = masks.astype(np.float32)
                    masks = torch.from_numpy(masks)
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(0), size=(ori_h, ori_w))
                    masks = masks.squeeze(0).detach().numpy()
                if masks.dtype != np.bool:
                    masks = masks >= 0.5
                segms_results = [[] for _ in range(len(self.CLASSES))]
                for j in range(len(dets)):
                    segms_results[labels[j]].append(masks[j])
                results.append((dets_results, segms_results))
            else:
                results.append(dets_results)
        return results


class ONNXRuntimeDetector(DeployBaseDetector):
    """Wrapper for detector's inference with ONNXRuntime."""

    def __init__(self, model_file, class_names, device_id, **kwargs):
        super(ONNXRuntimeDetector, self).__init__(class_names, device_id,
                                                  **kwargs)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(model_file, device_id)

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs[0]
        ort_outputs = self.model({'input': input_data})
        return ort_outputs


class TensorRTDetector(DeployBaseDetector):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, model_file, class_names, device_id, **kwargs):
        super(TensorRTDetector, self).__init__(class_names, device_id,
                                               **kwargs)
        from mmdeploy.apis.tensorrt import TRTWrapper

        self.model = TRTWrapper(model_file)
        self.output_names = ['dets', 'labels']
        if len(self.model.output_names) == 3:
            self.output_names.append('masks')

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs[0].contiguous()
        with torch.cuda.device(self.device_id), torch.no_grad():
            outputs = self.model({'input': input_data})
            outputs = [outputs[name] for name in self.output_names]
        outputs = [out.detach().cpu().numpy() for out in outputs]
        # filtered out invalid output filled with -1
        batch_labels = outputs[1]
        batch_size = batch_labels.shape[0]
        inds = batch_labels.reshape(-1) != -1
        for i in range(len(outputs)):
            ori_shape = outputs[i].shape
            outputs[i] = outputs[i].reshape(-1,
                                            *ori_shape[2:])[inds, ...].reshape(
                                                batch_size, -1, *ori_shape[2:])
        return outputs


class PPLDetector(DeployBaseDetector):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, model_file, class_names, device_id, **kwargs):
        super(PPLDetector, self).__init__(class_names, device_id)
        from mmdeploy.apis.ppl import PPLWrapper
        self.model = PPLWrapper(model_file, device_id)

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs[0]
        ppl_outputs = self.model({'input': input_data})
        return ppl_outputs


# Partition Single-Stage Base
class PartitionSingleStageDetector(DeployBaseDetector):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, class_names, model_cfg, deploy_cfg, device_id,
                 **kwargs):
        super(PartitionSingleStageDetector,
              self).__init__(class_names, device_id, **kwargs)
        # load cfg if necessary
        deploy_cfg = load_config(deploy_cfg)[0]
        model_cfg = load_config(model_cfg)[0]

        self.model_cfg = model_cfg
        self.deploy_cfg = deploy_cfg

    def partition0_postprocess(self, scores, bboxes):
        cfg = self.model_cfg.model.test_cfg
        deploy_cfg = self.deploy_cfg

        post_params = deploy_cfg.post_processing
        max_output_boxes_per_class = post_params.max_output_boxes_per_class
        iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
        score_threshold = cfg.get('score_thr', post_params.score_threshold)
        pre_top_k = post_params.pre_top_k
        keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
        return multiclass_nms(
            bboxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)


class ONNXRuntimePSSDetector(PartitionSingleStageDetector):
    """Wrapper for detector's inference with ONNXRuntime."""

    def __init__(self, model_file, class_names, model_cfg, deploy_cfg,
                 device_id, **kwargs):
        super(ONNXRuntimePSSDetector,
              self).__init__(class_names, model_cfg, deploy_cfg, device_id,
                             **kwargs)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(
            model_file, device_id, output_names=['scores', 'boxes'])

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs[0]
        ort_outputs = self.model({'input': input_data})
        scores, bboxes = ort_outputs[:2]
        scores = torch.from_numpy(scores).to(input_data.device)
        bboxes = torch.from_numpy(bboxes).to(input_data.device)
        return self.partition0_postprocess(scores, bboxes)


class NCNNPSSDetector(PartitionSingleStageDetector):
    """Wrapper for detector's inference with NCNN."""

    def __init__(self, model_file, class_names, model_cfg, deploy_cfg,
                 device_id, **kwargs):
        super(NCNNPSSDetector, self).__init__(class_names, model_cfg,
                                              deploy_cfg, device_id, **kwargs)
        from mmdeploy.apis.ncnn import NCNNWrapper
        assert len(model_file) == 2
        ncnn_param_file = model_file[0]
        ncnn_bin_file = model_file[1]
        self.model = NCNNWrapper(
            ncnn_param_file, ncnn_bin_file, output_names=['boxes', 'scores'])

    def forward_test(self, imgs, *args, **kwargs):
        imgs = imgs[0]

        outputs = self.model({'input': imgs})
        boxes = outputs['boxes']
        scores = outputs['scores']
        return self.partition0_postprocess(scores, boxes)


# Partition Two-Stage Base
class PartitionTwoStageDetector(DeployBaseDetector):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, class_names, model_cfg, deploy_cfg, device_id,
                 **kwargs):
        super(PartitionTwoStageDetector,
              self).__init__(class_names, device_id, **kwargs)
        from mmdet.models.builder import build_head, build_roi_extractor

        from mmdeploy.mmdet.models.roi_heads.bbox_heads import \
            get_bboxes_of_bbox_head

        # load cfg if necessary
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

        self.model_cfg = model_cfg
        self.deploy_cfg = deploy_cfg

        self.bbox_roi_extractor = build_roi_extractor(
            model_cfg.model.roi_head.bbox_roi_extractor)
        self.bbox_head = build_head(model_cfg.model.roi_head.bbox_head)

        class Context:
            pass

        ctx = Context()
        ctx.cfg = self.deploy_cfg
        self.get_bboxes_of_bbox_head = partial(get_bboxes_of_bbox_head, ctx)

    def partition0_postprocess(self, x, scores, bboxes):
        # rpn-nms + roi-extractor
        cfg = self.model_cfg.model.test_cfg.rpn
        deploy_cfg = self.deploy_cfg

        post_params = deploy_cfg.post_processing
        iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
        score_threshold = cfg.get('score_thr', post_params.score_threshold)
        pre_top_k = post_params.pre_top_k
        keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
        # only one class in rpn
        max_output_boxes_per_class = keep_top_k
        proposals, _ = multiclass_nms(
            bboxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)

        rois = proposals
        batch_index = torch.arange(
            rois.shape[0], device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        return rois, bbox_feats

    def partition1_postprocess(self, rois, cls_score, bbox_pred, img_metas):

        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))

        rcnn_test_cfg = self.model_cfg.model.test_cfg.rcnn
        return self.get_bboxes_of_bbox_head(self.bbox_head, rois, cls_score,
                                            bbox_pred,
                                            img_metas[0][0]['img_shape'],
                                            rcnn_test_cfg)


class ONNXRuntimePTSDetector(PartitionTwoStageDetector):
    """Wrapper for detector's inference with ONNXRuntime."""

    def __init__(self, model_file, class_names, model_cfg, deploy_cfg,
                 device_id, **kwargs):
        super(ONNXRuntimePTSDetector,
              self).__init__(class_names, model_cfg, deploy_cfg, device_id,
                             **kwargs)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model_list = [
            ORTWrapper(file, device_id=device_id) for file in model_file
        ]
        num_partition0_outputs = len(self.model_list[0].output_names)
        num_feat = num_partition0_outputs - 2
        self.model_list[0].output_names = [
            'feat/{}'.format(i) for i in range(num_feat)
        ] + ['scores', 'boxes']
        self.model_list[1].output_names = ['cls_score', 'bbox_pred']

    def forward_test(self, imgs, img_metas, *args, **kwargs):
        input_data = imgs[0]
        ort_outputs = self.model_list[0]({'input': input_data})
        feats = ort_outputs[:-2]
        scores, bboxes = ort_outputs[-2:]
        feats = [
            torch.from_numpy(feat).to(input_data.device) for feat in feats
        ]
        scores = torch.from_numpy(scores).to(input_data.device)
        bboxes = torch.from_numpy(bboxes).to(input_data.device)

        # partition0_postprocess
        rois, bbox_feats = self.partition0_postprocess(feats, scores, bboxes)

        # partition1
        ort_outputs = self.model_list[1]({'bbox_feats': bbox_feats})
        cls_score, bbox_pred = ort_outputs[:2]
        cls_score = torch.from_numpy(cls_score).to(input_data.device)
        bbox_pred = torch.from_numpy(bbox_pred).to(input_data.device)

        # partition1_postprocess
        return self.partition1_postprocess(rois, cls_score, bbox_pred,
                                           img_metas)


class TensorRTPTSDetector(PartitionTwoStageDetector):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, model_file, class_names, model_cfg, deploy_cfg,
                 device_id, **kwargs):
        super(TensorRTPTSDetector,
              self).__init__(class_names, model_cfg, deploy_cfg, device_id,
                             **kwargs)

        from mmdeploy.apis.tensorrt import TRTWrapper

        model_list = []
        for m_file in model_file:
            model = TRTWrapper(m_file)
            model_list.append(model)

        self.model_list = model_list

        output_names_list = []
        num_partition0_outputs = len(model_list[0].output_names)
        num_feat = num_partition0_outputs - 2
        output_names_list.append(
            ['feat/{}'.format(i)
             for i in range(num_feat)] + ['scores', 'boxes'])  # partition0
        output_names_list.append(['cls_score', 'bbox_pred'])  # partition1
        self.output_names_list = output_names_list

    def forward_test(self, imgs, img_metas, *args, **kwargs):

        # partition0 forward
        input_data = imgs[0].contiguous()
        with torch.cuda.device(self.device_id), torch.no_grad():
            outputs = self.model_list[0]({'input': input_data})
            outputs = [outputs[name] for name in self.output_names_list[0]]
        feats = outputs[:-2]
        scores, bboxes = outputs[-2:]

        # partition0_postprocess
        rois, bbox_feats = self.partition0_postprocess(feats, scores, bboxes)

        # partition1 forward
        bbox_feats = bbox_feats.contiguous()
        with torch.cuda.device(self.device_id), torch.no_grad():
            outputs = self.model_list[1]({'bbox_feats': bbox_feats})
            outputs = [outputs[name] for name in self.output_names_list[1]]
        cls_score, bbox_pred = outputs[:2]

        # partition1_postprocess
        outputs = self.partition1_postprocess(rois, cls_score, bbox_pred,
                                              img_metas)
        outputs = [out.detach().cpu() for out in outputs]
        return outputs


class NCNNPTSDetector(PartitionTwoStageDetector):
    """Wrapper for detector's inference with NCNN."""

    def __init__(self, model_file, class_names, model_cfg, deploy_cfg,
                 device_id, **kwargs):
        super(NCNNPTSDetector, self).__init__(class_names, model_cfg,
                                              deploy_cfg, device_id, **kwargs)
        from mmdeploy.apis.ncnn import NCNNWrapper
        assert self.device_id == -1
        assert len(model_file) == 4

        model_list = []
        for ncnn_param_file, ncnn_bin_file in zip(model_file[::2],
                                                  model_file[1::2]):
            model = NCNNWrapper(ncnn_param_file, ncnn_bin_file)
            model_list.append(model)

        model_cfg = load_config(model_cfg)[0]
        num_output_stage1 = model_cfg['model']['neck']['num_outs']

        output_names_list = []
        output_names_list.append(
            ['feat/{}'.format(i)
             for i in range(num_output_stage1)] + ['scores', 'boxes'])
        output_names_list.append(['cls_score', 'bbox_pred'])

        model_list[0].set_output_names(output_names_list[0])
        model_list[1].set_output_names(output_names_list[1])

        self.model_list = model_list
        self.output_names_list = output_names_list

    def forward_test(self, imgs, img_metas, *args, **kwargs):
        img = imgs[0]

        # stage0 forward
        out_stage0 = self.model_list[0]({'input': img})

        outputs = []
        for name in self.output_names_list[0]:
            out = out_stage0[name]
            outputs.append(out)
        feats = outputs[:-2]
        scores, bboxes = outputs[-2:]

        # stage0_postprocess
        rois, bbox_feats = self.partition0_postprocess(feats, scores, bboxes)

        # stage1 forward
        out_stage1 = self.model_list[1]({'bbox_feats': bbox_feats})
        cls_score = out_stage1['cls_score']
        bbox_pred = out_stage1['bbox_pred']

        # stage1_postprocess
        outputs = self.partition1_postprocess(rois, cls_score, bbox_pred,
                                              img_metas)
        outputs = [out.detach().cpu() for out in outputs]
        return outputs


def get_classes_from_config(model_cfg: Union[str, mmcv.Config], **kwargs):
    # load cfg if necessary
    model_cfg = load_config(model_cfg)[0]
    module_dict = DATASETS.module_dict
    data_cfg = model_cfg.data

    if 'train' in data_cfg:
        module = module_dict[data_cfg.train.type]
    elif 'val' in data_cfg:
        module = module_dict[data_cfg.val.type]
    elif 'test' in data_cfg:
        module = module_dict[data_cfg.test.type]
    else:
        raise RuntimeError(f'No dataset config found in: {model_cfg}')

    return module.CLASSES


ONNXRUNTIME_DETECTOR_MAP = dict(
    end2end=ONNXRuntimeDetector,
    single_stage_base=ONNXRuntimePSSDetector,
    two_stage_base=ONNXRuntimePTSDetector)
TENSORRT_DETECTOR_MAP = dict(
    end2end=TensorRTDetector, two_stage_base=TensorRTPTSDetector)

PPL_DETECTOR_MAP = dict(end2end=PPLDetector)
NCNN_DETECTOR_MAP = dict(
    single_stage_base=NCNNPSSDetector, two_stage_base=NCNNPTSDetector)

BACKEND_DETECTOR_MAP = {
    Backend.ONNXRUNTIME: ONNXRUNTIME_DETECTOR_MAP,
    Backend.TENSORRT: TENSORRT_DETECTOR_MAP,
    Backend.PPL: PPL_DETECTOR_MAP,
    Backend.NCNN: NCNN_DETECTOR_MAP
}


def build_detector(model_files, model_cfg, deploy_cfg, device_id, **kwargs):
    # load cfg if necessary
    deploy_cfg = load_config(deploy_cfg)[0]
    model_cfg = load_config(model_cfg)[0]

    backend = get_backend(deploy_cfg)
    class_names = get_classes_from_config(model_cfg)

    assert backend in BACKEND_DETECTOR_MAP, \
        f'Unsupported backend type: {backend.value}'
    detector_map = BACKEND_DETECTOR_MAP[backend]

    partition_type = 'end2end'
    if deploy_cfg.get('apply_marks', False):
        partition_params = deploy_cfg.get('partition_params', dict())
        partition_type = partition_params.get('partition_type', None)

    assert partition_type in detector_map,\
        f'Unsupported partition type: {partition_type}'
    backend_detector_class = detector_map[partition_type]

    model_files = model_files[0] if len(model_files) == 1 else model_files
    backend_detector = backend_detector_class(
        model_file=model_files,
        class_names=class_names,
        device_id=device_id,
        model_cfg=model_cfg,
        deploy_cfg=deploy_cfg,
        **kwargs)

    return backend_detector
