# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine import Config

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase, load_config
from mmdeploy.utils.test import WrapModel, check_backend, get_rewrite_outputs

try:
    import_codebase(Codebase.MMACTION)
except ImportError:
    pytest.skip(
        f'{Codebase.MMACTION} is not installed.', allow_module_level=True)


@pytest.mark.parametrize('backend', [Backend.ONNXRUNTIME])
@pytest.mark.parametrize('model_cfg_path',
                         ['tests/test_codebase/test_mmaction/data/model.py'])
def test_forward_of_base_recognizer(model_cfg_path, backend):
    check_backend(backend)
    deploy_cfg = Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(type='mmaction', task='VideoRecognition'),
            onnx_config=dict(
                type='onnx',
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                input_shape=None,
                input_names=['inputs'],
                output_names=['output'])))

    model_cfg = load_config(model_cfg_path)[0]
    from mmaction.apis import init_recognizer
    model = init_recognizer(model_cfg, None, device='cpu')

    img = torch.randn(1, 3, 3, 224, 224)
    from mmaction.structures import ActionDataSample
    data_sample = ActionDataSample()
    img_meta = dict(img_shape=(224, 224))
    data_sample.set_metainfo(img_meta)
    rewrite_inputs = {'inputs': img}
    wrapped_model = WrapModel(
        model, 'forward', data_samples=[data_sample], mode='predict')
    rewrite_outputs, _ = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    assert rewrite_outputs is not None
