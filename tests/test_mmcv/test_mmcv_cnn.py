# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch

from mmdeploy.utils import Backend
from mmdeploy.utils.test import check_backend, get_rewrite_outputs


def test_multiheadattention_ncnn():
    check_backend(Backend.NCNN)
    from mmcv.cnn.bricks.transformer import MultiheadAttention
    embed_dims, num_heads = 12, 2
    model = MultiheadAttention(embed_dims, num_heads, batch_first=True)
    query = torch.rand(1, 3, embed_dims)

    deploy_cfg = mmcv.Config(
        dict(
            onnx_config=dict(input_shape=None),
            backend_config=dict(type=Backend.NCNN.value),
        ))
    model_outputs = model(query)
    rewrite_inputs = dict(query=query)
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg,
        run_with_backend=True)
    if is_backend_output is None:
        assert rewrite_outputs is not None
    else:
        assert torch.allclose(
            model_outputs, rewrite_outputs[0], rtol=1e-03, atol=1e-05)
