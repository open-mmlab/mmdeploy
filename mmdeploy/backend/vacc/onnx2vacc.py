# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from typing import Optional, Union

import mmengine
import onnx
import tvm
import tvm.relay as relay
import vacc

from mmdeploy.utils import get_common_config, load_config


def from_onnx(onnx_model: str,
              output_path: str,
              model_input: dict,
              deploy_cfg: Union[str, mmengine.Config],
              dataset_file: Optional[str] = None,
              **kwargs):
    """Convert ONNX to VACC.

    Args:
        onnx_model (str): Input onnx model.
        output_path (str): File path to save VACC model.
        deploy_cfg (str | mmengine.Config): The path or content of config.
        model_cfg (str | mmengine.Config): The path or content of model config.
        dataset_file (str | None): The dataset file for quatization. Default to
            None.
    """

    # load deploy_cfg if necessary
    deploy_cfg = load_config(deploy_cfg)[0]

    if not isinstance(onnx_model, str):
        onnx_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
        onnx.save(onnx_model, onnx_path)
    else:
        onnx_path = onnx_model

    target = tvm.target.vacc()
    common_params = get_common_config(deploy_cfg)

    quant_mode = model_input.get('qconfig', {}).get('dtype', 'fp16')
    assert quant_mode in ['int8', 'fp16'], quant_mode + ' not support now'

    model = onnx.load(onnx_path)
    shape_dict = model_input['shape']
    mod, params = relay.frontend.from_onnx(model, shape_dict)

    func = mod['main']
    mod = relay.Module.from_expr(func)

    if quant_mode == 'int8':
        import random

        import h5py
        data = h5py.File(osp.join(output_path, 'calib_data.h5'),
                         'r')['calib_data']['input']
        calib_data = []

        index = list(range(len(data)))
        random.shuffle(index)
        for i in index[:1000]:
            calib_data.append({
                list(shape_dict.keys())[0]:
                tvm.nd.array(data[str(i)][:].astype('float32'))
            })

        with vacc.quantize.qconfig(
                calibrate_mode=model_input.get('qconfig',
                                               {}).get('calibrate_mode',
                                                       'percentile'),
                skip_conv_layers=model_input.get('qconfig', {}).get(
                    'skip_conv_layers', []),
                weight_scale=model_input.get('qconfig',
                                             {}).get('weight_scale', 'max'),
                quantize_per_channel=model_input.get('qconfig', {}).get(
                    'per_channel', False)):

            qmod = vacc.quantize.quantize(mod, params, calib_data)

        qmod = qmod['main']
        mod = relay.Module.from_expr(qmod)
        params = None

        data_type = 2
    else:
        data_type = 0

    with tvm.build_config(
            data_type=data_type,
            data_transport_mode=model_input.get('qconfig',
                                                {}).get('data_transmode', 1),
            mem_inplace=True,
            cluster_mode=model_input.get('qconfig', {}).get('cluster_mode',
                                                            0)):
        with relay.build_config(
                opt_level=2, stream_mode=True, enable_float_to_half=True):
            graph, lib, params = relay.build(
                mod=mod, target=target, params=params)

    save_dir = '-'.join([common_params['name'], quant_mode])
    model_name = common_params['name']
    output_root = osp.join(output_path, save_dir)
    if not osp.exists(output_root):
        os.makedirs(output_root)

    libpath = os.path.join(output_root, model_name + '.so')
    lib.export_library(libpath)

    graph_json_path = os.path.join(output_root, model_name + '.json')
    with open(graph_json_path, 'w') as f:
        f.write(graph)

    param_path = os.path.join(output_root, model_name + '.params')
    with open(param_path, 'wb') as f:
        f.write(relay.save_param_dict(params))

    assert osp.exists(os.path.join(output_root,
                                   model_name + '.params')), 'onnx2vacc failed'
    return [
        os.path.join(output_root, model_name + '.so'),
        os.path.join(output_root, model_name + '.json'),
        os.path.join(output_root, model_name + '.params')
    ]
