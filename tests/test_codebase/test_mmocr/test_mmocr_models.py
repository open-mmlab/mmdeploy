# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import mmengine
import numpy as np
import pytest
import torch

from mmdeploy.codebase import import_codebase
from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.config_utils import load_config
from mmdeploy.utils.test import (WrapModel, check_backend, get_backend_outputs,
                                 get_model_outputs, get_onnx_model,
                                 get_rewrite_outputs)

try:
    import_codebase(Codebase.MMOCR)
except ImportError:
    pytest.skip(f'{Codebase.MMOCR} is not installed.', allow_module_level=True)

from mmocr.models.textdet.necks import FPNC

dictionary = dict(
    type='Dictionary',
    dict_file='tests/test_codebase/test_mmocr/data/lower_english_digits.txt',
    with_padding=True,
    with_end=True)


class FPNCNeckModel(FPNC):

    def __init__(self, in_channels, init_cfg=None):
        super().__init__(in_channels, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.neck = FPNC(in_channels, init_cfg=init_cfg)

    def forward(self, inputs):
        neck_inputs = [
            inputs.repeat([1, channel, 1, 1]) for channel in self.in_channels
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
        type='FPNC', in_channels=[64, 128, 256, 512], lateral_channels=256)
    det_head = dict(
        type='DBHead',
        in_channels=256,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad'))
    model = SingleStageTextDetector(backbone, det_head, neck)

    model.requires_grad_(False)
    return model


def get_crnn_decoder_model(rnn_flag):
    from mmocr.models.textrecog.decoders import CRNNDecoder
    model = CRNNDecoder(32, dictionary, rnn_flag=rnn_flag)

    model.requires_grad_(False)
    return model


def get_fpnc_neck_model():
    model = FPNCNeckModel([2, 4, 8, 16])

    model.requires_grad_(False)
    return model


def get_base_recognizer_model():
    from mmocr.models.textrecog.recognizers import CRNN

    cfg = dict(
        preprocessor=None,
        backbone=dict(type='MiniVGG', leaky_relu=False, input_channels=1),
        encoder=None,
        decoder=dict(
            type='CRNNDecoder',
            in_channels=512,
            rnn_flag=True,
            module_loss=dict(type='CTCModuleLoss', letter_case='lower'),
            postprocessor=dict(type='CTCPostProcessor'),
            dictionary=dictionary),
        data_preprocessor=dict(
            type='mmocr.TextRecogDataPreprocessor', mean=[127], std=[127]))
    model = CRNN(
        backbone=cfg['backbone'],
        encoder=None,
        decoder=cfg['decoder'],
        data_preprocessor=cfg['data_preprocessor'])
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend', [Backend.NCNN])
def test_bidirectionallstm(backend: Backend):
    """Test forward rewrite of bidirectionallstm."""
    check_backend(backend)
    bilstm = get_bidirectionallstm_model()
    bilstm.cpu().eval()

    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(output_names=['output'], input_shape=None),
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
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=True)
    if is_backend_output:
        model_output = model_outputs.cpu().numpy()
        rewrite_output = rewrite_outputs[0].cpu().numpy()
        assert np.allclose(model_output, rewrite_output, rtol=1e-3, atol=1e-4)
    else:
        assert rewrite_outputs is not None


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test_nrtr_decoder__get_source_mask(backend: Backend):
    from mmocr.models.textrecog import NRTRDecoder
    deploy_cfg = mmengine.Config(
        dict(
            onnx_config=dict(
                input_names=['input'],
                output_names=['output'],
                input_shape=None,
                dynamic_axes={
                    'input': {
                        0: 'batch',
                    },
                    'output': {
                        0: 'batch',
                    }
                }),
            backend_config=dict(type=backend.value, model_inputs=None),
            codebase_config=dict(type='mmocr', task='TextRecognition')))
    src_seq = torch.rand(1, 200, 256)
    batch_src_seq = src_seq.expand(3, 200, 256)
    decoder = NRTRDecoder(
        dictionary=dict(
            type='Dictionary',
            dict_file='tests/test_codebase/test_mmocr/'
            'data/lower_english_digits.txt',
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=True))

    wrapped_model = WrapModel(decoder, '_get_source_mask')
    model_inputs = {'src_seq': src_seq, 'valid_ratios': torch.Tensor([1.0])}
    batch_model_inputs = {'input': batch_src_seq}
    ir_file_path = get_onnx_model(wrapped_model, model_inputs, deploy_cfg)
    backend_outputs = get_backend_outputs(ir_file_path, batch_model_inputs,
                                          deploy_cfg)[0].numpy()
    num_elements = np.prod(backend_outputs.shape[1:])
    # batch results should be same
    assert np.sum(backend_outputs[0] == backend_outputs[1]) == num_elements \
        and np.sum(backend_outputs[1] == backend_outputs[2]) == num_elements


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test_satrn_encoder__get_source_mask(backend: Backend):
    from mmocr.models.textrecog import SATRNEncoder

    deploy_cfg = mmengine.Config(
        dict(
            onnx_config=dict(
                input_names=['input'],
                output_names=['output'],
                input_shape=None,
                dynamic_axes={
                    'input': {
                        0: 'batch',
                    },
                    'output': {
                        0: 'batch',
                    }
                }),
            backend_config=dict(type=backend.value, model_inputs=None),
            codebase_config=dict(type='mmocr', task='TextRecognition')))
    encoder = SATRNEncoder(d_k=4, d_v=4, d_model=32, d_inner=32 * 4)
    feat = torch.randn(1, 32, 32, 32)
    batch_feat = feat.expand(3, 32, 32, 32)
    wrapped_model = WrapModel(encoder, 'forward')
    model_inputs = {'feat': feat}
    batch_model_inputs = {'input': batch_feat}
    ir_file_path = get_onnx_model(wrapped_model, model_inputs, deploy_cfg)
    backend_outputs = get_backend_outputs(ir_file_path, batch_model_inputs,
                                          deploy_cfg)[0].numpy()
    num_elements = np.prod(backend_outputs.shape[1:])
    # batch results should be same
    assert np.sum(backend_outputs[0] == backend_outputs[1]) == num_elements \
        and np.sum(backend_outputs[1] == backend_outputs[2]) == num_elements


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test_simple_test_of_single_stage_text_detector(backend: Backend):
    """Test simple_test single_stage_text_detector."""
    check_backend(backend)
    single_stage_text_detector = get_single_stage_text_detector_model()
    single_stage_text_detector.eval()

    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextDetection',
            )))

    input = torch.rand(1, 3, 64, 64)
    model_outputs = single_stage_text_detector._forward(input)

    wrapped_model = WrapModel(single_stage_text_detector, '_forward')
    rewrite_inputs = {'inputs': input}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=True)

    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0]

    model_outputs = model_outputs.cpu().numpy()
    rewrite_outputs = rewrite_outputs.cpu().numpy()
    assert np.allclose(model_outputs, rewrite_outputs, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('backend', [Backend.NCNN])
@pytest.mark.parametrize('rnn_flag', [True, False])
def test_crnndecoder(backend: Backend, rnn_flag: bool):
    """Test forward rewrite of crnndecoder."""
    check_backend(backend)
    crnn_decoder = get_crnn_decoder_model(rnn_flag)
    crnn_decoder.cpu().eval()

    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))

    input = torch.rand(1, 32, 1, 64)
    out_enc = None
    data_samples = None

    # to get outputs of pytorch model
    model_inputs = {
        'feat': input,
        'out_enc': out_enc,
        'data_samples': data_samples
    }
    model_outputs = get_model_outputs(crnn_decoder, 'forward_train',
                                      model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        crnn_decoder,
        'forward_train',
        out_enc=out_enc,
        data_samples=data_samples)
    rewrite_inputs = {'feat': input}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=True)
    rewrite_outputs = [rewrite_outputs[-1]]
    if is_backend_output:
        for model_output, rewrite_output in zip(model_outputs,
                                                rewrite_outputs):
            model_output = model_output.squeeze().cpu().numpy()
            rewrite_output = rewrite_output.squeeze()
            print(model_outputs, rewrite_output)
            assert np.allclose(
                model_output, rewrite_output, rtol=1e-03, atol=1e-04)
    else:
        assert rewrite_outputs is not None


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
@pytest.mark.parametrize(
    'data_samples', [[[{}]], [[{
        'resize_shape': [32, 32],
        'valid_ratio': 1.0
    }]]])
@pytest.mark.parametrize('is_dynamic', [True, False])
def test_forward_of_encoder_decoder_recognizer(data_samples, is_dynamic,
                                               backend):
    """Test forward base_recognizer."""
    check_backend(backend)
    base_recognizer = get_base_recognizer_model()
    base_recognizer.eval()

    if not is_dynamic:
        deploy_cfg = mmengine.Config(
            dict(
                backend_config=dict(type=backend.value),
                onnx_config=dict(input_shape=None),
                codebase_config=dict(
                    type='mmocr',
                    task='TextRecognition',
                )))
    else:
        deploy_cfg = mmengine.Config(
            dict(
                backend_config=dict(type=backend.value),
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

    model_outputs = base_recognizer.forward(input)
    wrapped_model = WrapModel(
        base_recognizer, 'forward', data_samples=data_samples[0])
    rewrite_inputs = {
        'batch_inputs': input,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)

    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0]

    model_outputs = model_outputs.cpu().numpy()
    rewrite_outputs = rewrite_outputs.cpu().numpy()
    assert np.allclose(model_outputs, rewrite_outputs, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('backend', [Backend.TENSORRT])
def test_forward_of_fpnc(backend: Backend):
    """Test forward rewrite of fpnc."""
    check_backend(backend)
    fpnc = get_fpnc_neck_model().cuda()
    fpnc.eval()
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(
                type=backend.value,
                common_config=dict(max_workspace_size=1 << 20),
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            inputs=dict(
                                min_shape=[1, 1, 64, 64],
                                opt_shape=[1, 1, 64, 64],
                                max_shape=[1, 1, 64, 64])))
                ]),
            onnx_config=dict(
                input_shape=None,
                input_names=['inputs'],
                output_names=['output']),
            codebase_config=dict(type='mmocr', task='TextDetection')))

    input = torch.rand(1, 1, 64, 64).cuda()
    model_inputs = {
        'inputs': input,
    }
    model_outputs = get_model_outputs(fpnc, 'forward', model_inputs)
    wrapped_model = WrapModel(fpnc, 'forward')
    rewrite_inputs = {
        'inputs': input,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        run_with_backend=False,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0]

    model_outputs = model_outputs.cpu().numpy()
    rewrite_outputs = rewrite_outputs.cpu().numpy()
    assert np.allclose(model_outputs, rewrite_outputs, rtol=1e-03, atol=1e-05)


def get_sar_model_cfg(decoder_type: str):
    model = dict(
        type='SARNet',
        data_preprocessor=dict(
            type='mmocr.TextRecogDataPreprocessor',
            mean=[127, 127, 127],
            std=[127, 127, 127]),
        backbone=dict(type='ResNet31OCR'),
        encoder=dict(
            type='mmocr.SAREncoder',
            enc_bi_rnn=False,
            enc_do_rnn=0.1,
            enc_gru=False),
        decoder=dict(
            type=f'mmocr.{decoder_type}',
            enc_bi_rnn=False,
            dec_bi_rnn=False,
            dec_do_rnn=0,
            dec_gru=False,
            pred_dropout=0.1,
            d_k=512,
            pred_concat=True,
            postprocessor=dict(type='AttentionPostprocessor'),
            module_loss=dict(
                type='CEModuleLoss', ignore_first_char=True, reduction='mean'),
            dictionary=dict(
                type='Dictionary',
                dict_file='tests/test_codebase/test_mmocr/'
                'data/lower_english_digits.txt',
                with_start=True,
                with_end=True,
                same_start_end=True,
                with_padding=True,
                with_unknown=True),
            max_seq_len=30))
    return mmengine.Config(dict(model=model))


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
@pytest.mark.parametrize('decoder_type',
                         ['SequentialSARDecoder', 'ParallelSARDecoder'])
def test_sar_model(backend: Backend, decoder_type):
    check_backend(backend)
    import os.path as osp

    import onnx
    from mmocr.models.textrecog import SARNet
    sar_cfg = get_sar_model_cfg(decoder_type)
    sar_cfg.model.pop('type')
    pytorch_model = SARNet(**(sar_cfg.model))

    model_inputs = {'inputs': torch.rand(1, 3, 48, 160), 'data_samples': None}

    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))
    # patch model
    pytorch_model.cfg = sar_cfg
    patched_model = patch_model(
        pytorch_model, cfg=deploy_cfg, backend=backend.value)
    onnx_file_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
    input_names = [k for k, v in model_inputs.items() if k != 'ctx']
    # model_forward = patched_model.forward
    # from functools import partial
    # patched_model.forward = partial(patched_model.forward,
    #                                 **{'data_samples': [data_sample]})
    with RewriterContext(
            cfg=deploy_cfg, backend=backend.value), torch.no_grad():
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


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test_mmdet_wrapper__forward(backend):
    check_backend(backend)
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    from mmocr.models.textdet import MMDetWrapper
    cfg, = load_config('tests/test_codebase/test_mmocr/data/mrcnn.py')

    model = MMDetWrapper(cfg.model.cfg)
    model.eval()
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(
                type=backend.value,
                common_config=dict(max_workspace_size=1 << 20)),
            onnx_config=dict(
                input_shape=None,
                input_names=['inputs'],
                output_names=['output']),
            codebase_config=dict(
                type='mmocr',
                task='TextDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    confidence_threshold=0.005,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=5000,
                    keep_top_k=100,
                    background_label_id=-1,
                    export_postprocess_mask=False))))

    input = torch.rand(1, 3, 64, 64)
    img_meta = {
        'ori_shape': [64, 64],
        'img_shape': [64, 64],
        'scale_factor': [1., 1.],
        'img_path': ''
    }
    pred_instances = InstanceData(metainfo=img_meta)
    data_sample = DetDataSample(pred_instances=pred_instances)
    data_sample.set_metainfo(img_meta)
    wrapped_model = WrapModel(model, 'forward', data_samples=[data_sample])

    rewrite_inputs = {'inputs': input}

    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=False)
    assert rewrite_outputs is not None


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test_abi_language_decoder___get_length(backend):
    check_backend(backend)
    from mmocr.models.textrecog.decoders import ABILanguageDecoder
    model = ABILanguageDecoder(dictionary=dictionary)
    input = torch.randn(1, 26, 37)
    model_inputs = {'logit': input}
    model_outputs = get_model_outputs(model, '_get_length', model_inputs)
    wrapped_model = WrapModel(model, '_get_length')
    rewrite_inputs = {'logit': input}
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0]

    model_outputs = model_outputs.float().cpu().numpy()
    rewrite_outputs = rewrite_outputs.cpu().numpy()
    print(model_outputs, rewrite_outputs)
    assert np.allclose(model_outputs, rewrite_outputs, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
def test__positional_encoding(backend):
    check_backend(backend)
    from mmocr.models.common.modules import PositionalEncoding
    pytorch_model = PositionalEncoding(64, 20)
    input = torch.rand(1, 20, 64)
    model_inputs = {'x': input}
    model_outputs = get_model_outputs(pytorch_model, 'forward', model_inputs)
    wrapped_model = WrapModel(pytorch_model, 'forward')
    rewrite_inputs = {'x': input}
    deploy_cfg = mmengine.Config(
        dict(
            backend_config=dict(type=backend.value),
            onnx_config=dict(input_shape=None),
            codebase_config=dict(
                type='mmocr',
                task='TextRecognition',
            )))
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        rewrite_outputs = rewrite_outputs[0]

    model_outputs = model_outputs.float().cpu().numpy()
    rewrite_outputs = rewrite_outputs.cpu().numpy()
    print(model_outputs, rewrite_outputs)
    assert np.allclose(model_outputs, rewrite_outputs, rtol=1e-03, atol=1e-05)
