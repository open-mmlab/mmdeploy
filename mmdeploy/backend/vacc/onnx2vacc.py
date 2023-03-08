# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Dict, Optional, Sequence


def from_onnx(
    onnx_model: str,
    output_path: str,
    model_name: str,
    input_shapes: Dict[str, Sequence],
    qconfig: Optional[Dict] = None,
):
    """Convert ONNX to VACC.

    Args:
        onnx_model (str): Input onnx model.
        output_path (str): File path to save VACC model.
        model_name (str): model name.
        input_shapes (ShapeType): The Default shape of the inputs.
        qconfig (Dict): Dictionary arguments feed to vacc.qconfig.
    """
    import onnx
    import tvm
    import tvm.relay as relay
    from vacc import quantize

    target = tvm.target.vacc()

    if qconfig is None:
        qconfig = dict()

    quant_mode = qconfig.get('dtype', 'fp16')
    assert quant_mode in ['int8', 'fp16'], quant_mode + ' not support now'

    shape_dict = input_shapes
    mod, params = relay.frontend.from_onnx(onnx.load(onnx_model), shape_dict)

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
        calib_num = qconfig.get('calib_num', 1000)
        for i in index[:calib_num]:
            calib_data.append({
                list(shape_dict.keys())[0]:
                tvm.nd.array(data[str(i)][:].astype('float32'))
            })

        with quantize.qconfig(
                calibrate_mode=qconfig.get('calibrate_mode', 'percentile'),
                skip_conv_layers=qconfig.get('skip_conv_layers', []),
                weight_scale=qconfig.get('weight_scale', 'max'),
                quantize_per_channel=qconfig.get('per_channel', False)):

            qmod = quantize.quantize(mod, params, calib_data)

        qmod = qmod['main']
        mod = relay.Module.from_expr(qmod)
        params = None

        data_type = 2
    else:
        data_type = 0

    with tvm.build_config(
            data_type=data_type,
            data_transport_mode=qconfig.get('data_transmode', 1),
            mem_inplace=True,
            cluster_mode=qconfig.get('cluster_mode', 0)):
        with relay.build_config(
                opt_level=2, stream_mode=True, enable_float_to_half=True):
            graph, lib, params = relay.build(
                mod=mod, target=target, params=params)

    save_dir = '-'.join([model_name, quant_mode])
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

    assert osp.exists(param_path), 'onnx2vacc failed'
    return [libpath, graph_json_path, param_path]
