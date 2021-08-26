import os.path as osp
import warnings
from functools import partial
from typing import Union

import mmcv
import numpy as np
import torch
from mmdet.core import bbox2result
from mmdet.models import BaseDetector

from mmdeploy.mmdet.core.post_processing import multiclass_nms


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

    def __init__(self, onnx_file, class_names, device_id, **kwargs):
        super(ONNXRuntimeDetector, self).__init__(class_names, device_id,
                                                  **kwargs)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(onnx_file, device_id)

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs[0]
        ort_outputs = self.model(input_data)
        return ort_outputs


class TensorRTDetector(DeployBaseDetector):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, model_file, class_names, device_id, **kwargs):
        super(TensorRTDetector, self).__init__(class_names, device_id,
                                               **kwargs)
        from mmdeploy.apis.tensorrt import TRTWrapper, load_tensorrt_plugin
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom plugins, \
                you may have to build backend ops with TensorRT')
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


# Split Single-Stage Base
class SplitSingleStageBaseDetector(DeployBaseDetector):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, class_names, model_cfg, deploy_cfg, device_id,
                 **kwargs):
        super().__init__(class_names, device_id, **kwargs)
        # load deploy_cfg if necessary
        if isinstance(deploy_cfg, str):
            deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
        if not isinstance(deploy_cfg, mmcv.Config):
            raise TypeError('deploy_cfg must be a filename or Config object, '
                            f'but got {type(deploy_cfg)}')

        # load model_cfg if needed
        if isinstance(model_cfg, str):
            model_cfg = mmcv.Config.fromfile(model_cfg)
        if not isinstance(model_cfg, mmcv.Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(model_cfg)}')

        self.model_cfg = model_cfg
        self.deploy_cfg = deploy_cfg

    def split0_postprocess(self, scores, bboxes):
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


class ONNXRuntimeSSSBDetector(SplitSingleStageBaseDetector):
    """Wrapper for detector's inference with ONNXRuntime."""

    def __init__(self, model_file, class_names, model_cfg, deploy_cfg,
                 device_id, **kwargs):
        super().__init__(class_names, model_cfg, deploy_cfg, device_id,
                         **kwargs)
        import onnxruntime as ort

        # get the custom op path
        from mmdeploy.apis.onnxruntime import get_ops_path
        ort_custom_op_path = get_ops_path()
        session_options = ort.SessionOptions()
        # register custom op for onnxruntime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = ort.InferenceSession(model_file, session_options)
        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})

        sess.set_providers(providers, options)

        self.sess = sess
        self.io_binding = sess.io_binding()
        self.output_names = ['scores', 'boxes']
        self.is_cuda_available = is_cuda_available

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs[0]
        # set io binding for inputs/outputs
        device_type = 'cuda' if self.is_cuda_available else 'cpu'
        if not self.is_cuda_available:
            input_data = input_data.cpu()
        self.io_binding.bind_input(
            name='input',
            device_type=device_type,
            device_id=self.device_id,
            element_type=np.float32,
            shape=input_data.shape,
            buffer_ptr=input_data.data_ptr())

        for name in self.output_names:
            self.io_binding.bind_output(name)
        # run session to get outputs
        self.sess.run_with_iobinding(self.io_binding)
        ort_outputs = self.io_binding.copy_outputs_to_cpu()
        scores, bboxes = ort_outputs[:2]
        scores = torch.from_numpy(scores).to(input_data.device)
        bboxes = torch.from_numpy(bboxes).to(input_data.device)
        return self.split0_postprocess(scores, bboxes)


class NCNNSSSBDetector(SplitSingleStageBaseDetector):
    """Wrapper for detector's inference with NCNN."""

    def __init__(self, model_file, class_names, model_cfg, deploy_cfg,
                 device_id, **kwargs):
        super().__init__(class_names, model_cfg, deploy_cfg, device_id,
                         **kwargs)
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
        return self.split0_postprocess(scores, boxes)


# Split Two-Stage Base
class SplitTwoStageBaseDetector(DeployBaseDetector):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, class_names, model_cfg, deploy_cfg, device_id,
                 **kwargs):
        super().__init__(class_names, device_id, **kwargs)
        from mmdet.models.builder import build_roi_extractor, build_head
        from mmdeploy.mmdet.models.roi_heads.bbox_heads import \
            get_bboxes_of_bbox_head

        # load deploy_cfg if necessary
        if isinstance(deploy_cfg, str):
            deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
        if not isinstance(deploy_cfg, mmcv.Config):
            raise TypeError('deploy_cfg must be a filename or Config object, '
                            f'but got {type(deploy_cfg)}')

        # load model_cfg if needed
        if isinstance(model_cfg, str):
            model_cfg = mmcv.Config.fromfile(model_cfg)
        if not isinstance(model_cfg, mmcv.Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(model_cfg)}')

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

    def split0_postprocess(self, x, scores, bboxes):
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

    def split1_postprocess(self, rois, cls_score, bbox_pred, img_metas):

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


class ONNXRuntimeSTSBDetector(SplitTwoStageBaseDetector):
    """Wrapper for detector's inference with ONNXRuntime."""

    def __init__(self, model_file, class_names, model_cfg, deploy_cfg,
                 device_id, **kwargs):
        super().__init__(class_names, model_cfg, deploy_cfg, device_id,
                         **kwargs)
        import onnxruntime as ort

        # get the custom op path
        from mmdeploy.apis.onnxruntime import get_ops_path
        ort_custom_op_path = get_ops_path()
        session_options = ort.SessionOptions()
        # register custom op for onnxruntime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})

        sess_list = []
        io_binding_list = []
        for m_file in model_file:
            sess = ort.InferenceSession(m_file, session_options)
            sess.set_providers(providers, options)
            sess_list.append(sess)
            io_binding_list.append(sess.io_binding())

        self.sess_list = sess_list
        self.io_binding_list = io_binding_list

        output_names_list = []
        num_split0_outputs = len(sess_list[0].get_outputs())
        num_feat = num_split0_outputs - 2
        output_names_list.append(
            ['feat/{}'.format(i)
             for i in range(num_feat)] + ['scores', 'boxes'])  # split0
        output_names_list.append(['cls_score', 'bbox_pred'])  # split1
        self.output_names_list = output_names_list

        self.is_cuda_available = is_cuda_available

    def forward_test(self, imgs, img_metas, *args, **kwargs):
        input_data = imgs[0]
        # set io binding for inputs/outputs
        device_type = 'cuda' if self.is_cuda_available else 'cpu'

        # split0
        if not self.is_cuda_available:
            input_data = input_data.cpu()
        self.io_binding_list[0].bind_input(
            name='input',
            device_type=device_type,
            device_id=self.device_id,
            element_type=np.float32,
            shape=input_data.shape,
            buffer_ptr=input_data.data_ptr())

        for name in self.output_names_list[0]:
            self.io_binding_list[0].bind_output(name)
        # run session to get outputs
        self.sess_list[0].run_with_iobinding(self.io_binding_list[0])
        ort_outputs = self.io_binding_list[0].copy_outputs_to_cpu()
        feats = ort_outputs[:-2]
        scores, bboxes = ort_outputs[-2:]
        feats = [
            torch.from_numpy(feat).to(input_data.device) for feat in feats
        ]
        scores = torch.from_numpy(scores).to(input_data.device)
        bboxes = torch.from_numpy(bboxes).to(input_data.device)

        # split0_postprocess
        rois, bbox_feats = self.split0_postprocess(feats, scores, bboxes)

        # split1
        if not self.is_cuda_available:
            bbox_feats = bbox_feats.cpu()
        self.io_binding_list[1].bind_input(
            name='bbox_feats',
            device_type=device_type,
            device_id=self.device_id,
            element_type=np.float32,
            shape=bbox_feats.shape,
            buffer_ptr=bbox_feats.data_ptr())

        for name in self.output_names_list[1]:
            self.io_binding_list[1].bind_output(name)
        # run session to get outputs
        self.sess_list[1].run_with_iobinding(self.io_binding_list[1])
        ort_outputs = self.io_binding_list[1].copy_outputs_to_cpu()
        cls_score, bbox_pred = ort_outputs[:2]
        cls_score = torch.from_numpy(cls_score).to(input_data.device)
        bbox_pred = torch.from_numpy(bbox_pred).to(input_data.device)

        # split1_postprocess
        return self.split1_postprocess(rois, cls_score, bbox_pred, img_metas)


class PPLDetector(DeployBaseDetector):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, model_file, class_names, device_id, **kwargs):
        super(PPLDetector, self).__init__(class_names, device_id)
        import pyppl.nn as pplnn
        from mmdeploy.apis.ppl import register_engines

        # enable quick select by default to speed up pipeline
        # TODO: open it to users after ppl supports saving serialized models
        # TODO: disable_avx512 will be removed or open to users in config
        engines = register_engines(
            device_id, disable_avx512=False, quick_select=True)
        cuda_options = pplnn.CudaEngineOptions()
        cuda_options.device_id = device_id
        runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(
            model_file, engines)
        assert runtime_builder is not None, 'Failed to create '\
            'OnnxRuntimeBuilder.'

        runtime_options = pplnn.RuntimeOptions()
        runtime = runtime_builder.CreateRuntime(runtime_options)
        assert runtime is not None, 'Failed to create the instance of Runtime.'

        self.runtime = runtime
        self.CLASSES = class_names
        self.device_id = device_id
        self.inputs = [
            runtime.GetInputTensor(i) for i in range(runtime.GetInputCount())
        ]

    def forward_test(self, imgs, *args, **kwargs):
        import pyppl.common as pplcommon
        input_data = imgs[0].contiguous()
        self.inputs[0].ConvertFromHost(input_data.cpu().numpy())
        status = self.runtime.Run()
        assert status == pplcommon.RC_SUCCESS, 'Run() '\
            'failed: ' + pplcommon.GetRetCodeStr(status)
        status = self.runtime.Sync()
        assert status == pplcommon.RC_SUCCESS, 'Sync() '\
            'failed: ' + pplcommon.GetRetCodeStr(status)
        outputs = []
        for i in range(self.runtime.GetOutputCount()):
            out_tensor = self.runtime.GetOutputTensor(i).ConvertToHost()
            outputs.append(np.array(out_tensor, copy=False))
        return outputs


class TensorRTSTSBDetector(SplitTwoStageBaseDetector):
    """Wrapper for detector's inference with TensorRT."""

    def __init__(self, model_file, class_names, model_cfg, deploy_cfg,
                 device_id, **kwargs):
        super().__init__(class_names, model_cfg, deploy_cfg, device_id,
                         **kwargs)

        from mmdeploy.apis.tensorrt import TRTWrapper, load_tensorrt_plugin
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom plugins, \
                you may have to build backend ops with TensorRT')

        model_list = []
        for m_file in model_file:
            model = TRTWrapper(m_file)
            model_list.append(model)

        self.model_list = model_list

        output_names_list = []
        num_split0_outputs = len(model_list[0].output_names)
        num_feat = num_split0_outputs - 2
        output_names_list.append(
            ['feat/{}'.format(i)
             for i in range(num_feat)] + ['scores', 'boxes'])  # split0
        output_names_list.append(['cls_score', 'bbox_pred'])  # split1
        self.output_names_list = output_names_list

    def forward_test(self, imgs, img_metas, *args, **kwargs):

        # split0 forward
        input_data = imgs[0].contiguous()
        with torch.cuda.device(self.device_id), torch.no_grad():
            outputs = self.model_list[0]({'input': input_data})
            outputs = [outputs[name] for name in self.output_names_list[0]]
        feats = outputs[:-2]
        scores, bboxes = outputs[-2:]

        # split0_postprocess
        rois, bbox_feats = self.split0_postprocess(feats, scores, bboxes)

        # split1 forward
        bbox_feats = bbox_feats.contiguous()
        with torch.cuda.device(self.device_id), torch.no_grad():
            outputs = self.model_list[1]({'bbox_feats': bbox_feats})
            outputs = [outputs[name] for name in self.output_names_list[1]]
        cls_score, bbox_pred = outputs[:2]

        # split1_postprocess
        outputs = self.split1_postprocess(rois, cls_score, bbox_pred,
                                          img_metas)
        outputs = [out.detach().cpu() for out in outputs]
        return outputs


class NCNNSTSBDetector(SplitTwoStageBaseDetector):
    """Wrapper for detector's inference with NCNN."""

    def __init__(self, model_file, class_names, model_cfg, deploy_cfg,
                 device_id, **kwargs):
        super().__init__(class_names, model_cfg, deploy_cfg, device_id,
                         **kwargs)
        from mmdeploy.apis.ncnn import NCNNWrapper
        assert self.device_id == -1
        assert len(model_file) == 4

        model_list = []
        for ncnn_param_file, ncnn_bin_file in zip(model_file[::2],
                                                  model_file[1::2]):
            model = NCNNWrapper(ncnn_param_file, ncnn_bin_file)
            model_list.append(model)

        # TODO: update this after refactor
        if isinstance(model_cfg, str):
            model_cfg = mmcv.Config.fromfile(model_cfg)
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
        rois, bbox_feats = self.split0_postprocess(feats, scores, bboxes)

        # stage1 forward
        out_stage1 = self.model_list[1]({'bbox_feats': bbox_feats})
        cls_score = out_stage1['cls_score']
        bbox_pred = out_stage1['bbox_pred']

        # stage1_postprocess
        outputs = self.split1_postprocess(rois, cls_score, bbox_pred,
                                          img_metas)
        outputs = [out.detach().cpu() for out in outputs]
        return outputs


def get_classes_from_config(model_cfg: Union[str, mmcv.Config], **kwargs):
    if isinstance(model_cfg, str):
        model_cfg = mmcv.Config.fromfile(model_cfg)
    elif not isinstance(model_cfg, (mmcv.Config, mmcv.ConfigDict)):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(model_cfg)}')

    from mmdet.datasets import DATASETS
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
    single_stage_base=ONNXRuntimeSSSBDetector,
    two_stage_base=ONNXRuntimeSTSBDetector)
TENSORRT_DETECTOR_MAP = dict(
    end2end=TensorRTDetector, two_stage_base=TensorRTSTSBDetector)

PPL_DETECTOR_MAP = dict(end2end=PPLDetector)
NCNN_DETECTOR_MAP = dict(
    single_stage_base=NCNNSSSBDetector, two_stage_base=NCNNSTSBDetector)

BACKEND_DETECTOR_MAP = dict(
    onnxruntime=ONNXRUNTIME_DETECTOR_MAP,
    tensorrt=TENSORRT_DETECTOR_MAP,
    ppl=PPL_DETECTOR_MAP,
    ncnn=NCNN_DETECTOR_MAP)


def build_detector(model_files, model_cfg, deploy_cfg, device_id, **kwargs):

    if isinstance(model_cfg, str):
        model_cfg = mmcv.Config.fromfile(model_cfg)
    elif not isinstance(model_cfg, (mmcv.Config, mmcv.ConfigDict)):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(model_cfg)}')

    if isinstance(deploy_cfg, str):
        deploy_cfg = mmcv.Config.fromfile(deploy_cfg)
    elif not isinstance(deploy_cfg, (mmcv.Config, mmcv.ConfigDict)):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(deploy_cfg)}')

    backend = deploy_cfg['backend']
    class_names = get_classes_from_config(model_cfg)

    assert backend in BACKEND_DETECTOR_MAP, \
        f'Unsupported backend type: {backend}'
    detector_map = BACKEND_DETECTOR_MAP[backend]

    split_type = 'end2end'
    if deploy_cfg.get('apply_marks', False):
        split_params = deploy_cfg.get('split_params', dict())
        split_type = split_params.get('split_type', None)

    assert split_type in detector_map, f'Unsupported split type: {split_type}'
    backend_detector_class = detector_map[split_type]

    model_files = model_files[0] if len(model_files) == 1 else model_files
    backend_detector = backend_detector_class(
        model_file=model_files,
        class_names=class_names,
        device_id=device_id,
        model_cfg=model_cfg,
        deploy_cfg=deploy_cfg,
        **kwargs)

    return backend_detector
