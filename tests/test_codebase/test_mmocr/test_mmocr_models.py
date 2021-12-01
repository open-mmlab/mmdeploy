# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import mmcv
import numpy as np
import pytest
import torch
from mmocr.models.textdet.necks import FPNC

from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import Backend
from mmdeploy.utils.test import (WrapModel, check_backend, get_model_outputs,
                                 get_rewrite_outputs)


class FPNCNeckModel(FPNC):

    def __init__(self, in_channels, init_cfg=None):
        super().__init__(in_channels, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.neck = FPNC(in_channels, init_cfg=init_cfg)

    def forward(self, inputs):
        neck_inputs = [
            torch.ones(1, channel, inputs.shape[-2], inputs.shape[-1])
            for channel in self.in_channels
        ]
        output = self.neck.forward(neck_inputs)
        return output


def get_bidirectionallstm_model():
    from mmocr.models.textrecog.layers.lstm_layer import BidirectionalLSTM
    model = BidirectionalLSTM(32, 16, 16)

    model.requires_grad_(False)
    return model


def get_single_stage_text_detector_model():
    from mmocr.models.textdet import SingleStageTextDetector
    backbone = dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=False,
        style='caffe')
    neck = dict(
        type='FPNC',
        in_channels=[64, 128, 256, 512],
        lateral_channels=4,
        out_channels=4)
    bbox_head = dict(
        type='DBHead',
        text_repr_type='quad',
        in_channels=16,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True))
    model = SingleStageTextDetector(backbone, neck, bbox_head)

    model.requires_grad_(False)
    return model


def get_encode_decode_recognizer_model():
    from mmocr.models.textrecog import EncodeDecodeRecognizer

    cfg = dict(
        preprocessor=None,
        backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
        encoder=dict(type='TFEncoder'),
        decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
        loss=dict(type='CTCLoss'),
        label_convertor=dict(
            type='CTCConvertor',
            dict_type='DICT36',
            with_unknown=False,
            lower=True),
        pretrained=None)

    model = EncodeDecodeRecognizer(
        backbone=cfg['backbone'],
        encoder=cfg['encoder'],
        decoder=cfg['decoder'],
        loss=cfg['loss'],
        label_convertor=cfg['label_convertor'])
    model.requires_grad_(False)
    return model


def get_crnn_decoder_model(rnn_flag):
    from mmocr.models.textrecog.decoders import CRNNDecoder
    model = CRNNDecoder(32, 4, rnn_flag=rnn_flag)

    model.requires_grad_(False)
    return model


def get_fpnc_neck_model():
    model = FPNCNeckModel([2, 4, 8, 16])

    model.requires_grad_(False)
    return model


def get_base_recognizer_model():
    from mmocr.models.textrecog import CRNNNet

    cfg = dict(
        preprocessor=None,
        backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
        encoder=None,
        decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
        loss=dict(type='CTCLoss'),
        label_convertor=dict(
            type='CTCConvertor',
            dict_type='DICT36',
            with_unknown=False,
            lower=True),
        pretrained=None)

    model = CRNNNet(
        backbone=cfg['backbone'],
        decoder=cfg['decoder'],
        loss=cfg['loss'],
        label_convertor=cfg['label_convertor'])
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.NCNN])
def test_bidirectionallstm(backend_type: Backend):
    """Test forward rewrite of bidirectionallstm."""
    check_backend(backend_type)
    bilstm = get_bidirectionallstm_model()
    bilstm.cpu().eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))

    input = torch.rand(1, 1, 32)

    # to get outputs of pytorch model
    model_inputs = {
        'input': input,
    }
    model_outputs = get_model_outputs(bilstm, 'forward', model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(bilstm, 'forward')
    rewrite_inputs = {'input': input}
    rewrite_outputs = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        rewrite_output = rewrite_output.squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


def test_simple_test_of_single_stage_text_detector():
    """Test simple_test single_stage_text_detector."""
    single_stage_text_detector = get_single_stage_text_detector_model()
    single_stage_text_detector.eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='default'),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextDetection',
            )))

    input = torch.rand(1, 3, 64, 64)
    img_metas = [{
        'ori_shape': [64, 64, 3],
        'img_shape': [64, 64, 3],
        'pad_shape': [64, 64, 3],
        'scale_factor': [1., 1., 1., 1],
    }]

    x = single_stage_text_detector.extract_feat(input)
    model_outputs = single_stage_text_detector.bbox_head(x)

    wrapped_model = WrapModel(single_stage_text_detector, 'simple_test')
    rewrite_inputs = {'img': input, 'img_metas': img_metas[0]}
    rewrite_outputs = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        rewrite_output = rewrite_output.squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('backend_type', [Backend.NCNN])
@pytest.mark.parametrize('rnn_flag', [True, False])
def test_crnndecoder(backend_type: Backend, rnn_flag: bool):
    """Test forward rewrite of crnndecoder."""
    check_backend(backend_type)
    crnn_decoder = get_crnn_decoder_model(rnn_flag)
    crnn_decoder.cpu().eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))

    input = torch.rand(1, 32, 1, 64)
    out_enc = None
    targets_dict = None
    img_metas = None

    # to get outputs of pytorch model
    model_inputs = {
        'feat': input,
        'out_enc': out_enc,
        'targets_dict': targets_dict,
        'img_metas': img_metas
    }
    model_outputs = get_model_outputs(crnn_decoder, 'forward_train',
                                      model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        crnn_decoder,
        'forward_train',
        out_enc=out_enc,
        targets_dict=targets_dict,
        img_metas=img_metas)
    rewrite_inputs = {'feat': input}
    rewrite_outputs = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        rewrite_output = rewrite_output.squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize(
    'img_metas', [[[{}]], [[{
        'resize_shape': [32, 32],
        'valid_ratio': 1.0
    }]]])
@pytest.mark.parametrize('is_dynamic', [True, False])
def test_forward_of_base_recognizer(img_metas, is_dynamic):
    """Test forward base_recognizer."""
    base_recognizer = get_base_recognizer_model()
    base_recognizer.eval()

    if not is_dynamic:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type='ncnn'),
                onnx_config=dict(input_shape=None),
                codebase_config=dict(
                    type='mmocr',
                    task='TextRecognition',
                )))
    else:
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(type='ncnn'),
                onnx_config=dict(
                    input_shape=None,
                    dynamic_axes={
                        'input': {
                            0: 'batch',
                            2: 'height',
                            3: 'width'
                        },
                        'output': {
                            0: 'batch',
                            2: 'height',
                            3: 'width'
                        }
                    }),
                codebase_config=dict(
                    type='mmocr',
                    task='TextRecognition',
                )))

    input = torch.rand(1, 1, 32, 32)

    feat = base_recognizer.extract_feat(input)
    out_enc = None
    if base_recognizer.encoder is not None:
        out_enc = base_recognizer.encoder(feat, img_metas)
    model_outputs = base_recognizer.decoder(
        feat, out_enc, None, img_metas, train_mode=False)
    wrapped_model = WrapModel(
        base_recognizer, 'forward', img_metas=img_metas[0])
    rewrite_inputs = {
        'img': input,
    }
    rewrite_outputs = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        rewrite_output = rewrite_output.squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


def test_simple_test_of_encode_decode_recognizer():
    """Test simple_test encode_decode_recognizer."""
    encode_decode_recognizer = get_encode_decode_recognizer_model()
    encode_decode_recognizer.eval()

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='default'),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))

    input = torch.rand(1, 1, 32, 32)
    img_metas = [{'resize_shape': [32, 32], 'valid_ratio': 1.0}]

    feat = encode_decode_recognizer.extract_feat(input)
    out_enc = None
    if encode_decode_recognizer.encoder is not None:
        out_enc = encode_decode_recognizer.encoder(feat, img_metas)
    model_outputs = encode_decode_recognizer.decoder(
        feat, out_enc, None, img_metas, train_mode=False)

    wrapped_model = WrapModel(
        encode_decode_recognizer, 'simple_test', img_metas=img_metas)
    rewrite_inputs = {'img': input}
    rewrite_outputs = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    for model_output, rewrite_output in zip(model_outputs, rewrite_outputs):
        model_output = model_output.squeeze().cpu().numpy()
        rewrite_output = rewrite_output.squeeze()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('backend_type', [Backend.TENSORRT])
def test_forward_of_fpnc(backend_type: Backend):
    """Test forward rewrite of fpnc."""
    check_backend(backend_type)
    fpnc = get_fpnc_neck_model()
    fpnc.eval()
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(
                type=backend_type.value,
                common_config=dict(max_workspace_size=1 << 30),
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            input=dict(
                                min_shape=[1, 3, 64, 64],
                                opt_shape=[1, 3, 64, 64],
                                max_shape=[1, 3, 64, 64])))
                ]),
            onnx_config=dict(input_shape=[64, 64], output_names=['output']),
            codebase_config=dict(type='mmocr', task='TextDetection')))

    input = torch.rand(1, 3, 64, 64).cuda()
    model_inputs = {
        'inputs': input,
    }
    model_outputs = get_model_outputs(fpnc, 'forward', model_inputs)
    wrapped_model = WrapModel(fpnc, 'forward')
    rewrite_inputs = {
        'inputs': input,
    }
    rewrite_outputs, is_need_name = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    if is_need_name:
        model_output = model_outputs[0].squeeze().cpu().numpy()
        rewrite_output = rewrite_outputs[0].squeeze().cpu().numpy()
        assert np.allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
    else:
        for model_output, rewrite_output in zip(model_outputs,
                                                rewrite_outputs):
            model_output = model_output.squeeze().cpu().numpy()
            rewrite_output = rewrite_output.squeeze().cpu().numpy()
            assert np.allclose(
                model_output, rewrite_output, rtol=1e-03, atol=1e-05)


def get_sar_model_cfg(decoder_type: str):
    label_convertor = dict(
        type='AttnConvertor', dict_type='DICT90', with_unknown=True)

    model = dict(
        type='SARNet',
        backbone=dict(type='ResNet31OCR'),
        encoder=dict(
            type='SAREncoder',
            enc_bi_rnn=False,
            enc_do_rnn=0.1,
            enc_gru=False,
        ),
        decoder=dict(
            type=decoder_type,
            enc_bi_rnn=False,
            dec_bi_rnn=False,
            dec_do_rnn=0,
            dec_gru=False,
            pred_dropout=0.1,
            d_k=512,
            pred_concat=True),
        loss=dict(type='SARLoss'),
        label_convertor=label_convertor,
        max_seq_len=30)
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiRotateAugOCR',
            rotate_degrees=[0, 90, 270],
            transforms=[
                dict(
                    type='ResizeOCR',
                    height=48,
                    min_width=48,
                    max_width=160,
                    keep_aspect_ratio=True,
                    width_downsample_ratio=0.25),
                dict(type='ToTensorOCR'),
                dict(
                    type='Collect',
                    keys=['img'],
                    meta_keys=[
                        'filename', 'ori_shape', 'resize_shape', 'valid_ratio'
                    ]),
            ])
    ]
    return mmcv.Config(
        dict(model=model, data=dict(test=dict(pipeline=test_pipeline))))


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
@pytest.mark.parametrize('decoder_type',
                         ['SequentialSARDecoder', 'ParallelSARDecoder'])
def test_sar_model(backend_type: Backend, decoder_type):
    check_backend(backend_type)
    import os.path as osp
    import onnx
    from mmocr.models.textrecog import SARNet
    sar_cfg = get_sar_model_cfg(decoder_type)
    sar_cfg.model.pop('type')
    pytorch_model = SARNet(**(sar_cfg.model))
    model_inputs = {'x': torch.rand(1, 3, 48, 160)}

    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))
    # patch model
    pytorch_model.cfg = sar_cfg
    patched_model = patch_model(
        pytorch_model, cfg=deploy_cfg, backend=backend_type.value)
    onnx_file_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
    input_names = [k for k, v in model_inputs.items() if k != 'ctx']
    with RewriterContext(
            cfg=deploy_cfg, backend=backend_type.value), torch.no_grad():
        torch.onnx.export(
            patched_model,
            tuple([v for k, v in model_inputs.items()]),
            onnx_file_path,
            export_params=True,
            input_names=input_names,
            output_names=None,
            opset_version=11,
            dynamic_axes=None,
            keep_initializers_as_inputs=False)

    # The result should be different due to the rewrite.
    # So we only check if the file exists
    assert osp.exists(onnx_file_path)

    model = onnx.load(onnx_file_path)
    assert model is not None
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        assert False
