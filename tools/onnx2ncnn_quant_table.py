# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import math

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_root_logger, load_config


def gen_quant_table(onnx_path: str,
                    deploy_cfg_path: str,
                    model_cfg_path: str,
                    output_quant_table_path: str,
                    output_onnx_path: str,
                    image_dir: str = None,
                    device: str = 'cuda',
                    dataset_type: str = 'val',
                    debug: bool = True):
    from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
    from ppq.api import export_ppq_graph, quantize_onnx_model

    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # build calibration data load Iterable

    task_processor = build_task_processor(model_cfg, deploy_cfg, device)
    import pdb
    pdb.set_trace()
    dataset = task_processor.build_dataset(model_cfg, dataset_type)
        
    dataloader = task_processor.build_dataloader(
        dataset, 1, 1, shuffle=not debug)

    input_shape = deploy_cfg.input_shape
    if input_shape is None:
        # get an available input shape randomly
        for _, input_data in enumerate(dataloader):
            input_shape = input_data['img'].shape
            break

    collate_fn = None
    if image_dir is None:
        collate_fn = lambda x: x['img'].to(device)
    else:
        pass

    # settings for ncnn quantization
    quant_setting = QuantizationSettingFactory.default_setting()
    quant_setting.equalization = False
    quant_setting.dispatcher = 'conservative'

    # quantize the model
    quantized = quantize_onnx_model(
        onnx_import_file=onnx_path,
        calib_dataloader=dataloader,
        calib_steps=512,
        input_shape=input_shape,
        setting=quant_setting,
        collate_fn=collate_fn,
        platform=TargetPlatform.NCNN_INT8,
        device=device,
        verbose=1)

    # export quantized graph and quant table
    export_ppq_graph(
        graph=quantized,
        platform=TargetPlatform.NCNN_INT8,
        graph_save_to=output_onnx_path,
        config_save_to=output_quant_table_path)
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX to ncnn.')
    parser.add_argument('onnx_path', help='ONNX model path')
    parser.add_argument('deploy_cfg_path', help='input deploy config path')
    parser.add_argument('model_cfg_path', help='input model config path')
    parser.add_argument('quant_table_path', help='output quant table path')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger = get_root_logger(log_level=args.log_level)

    onnx_path = args.onnx_path
    deploy_cfg_path = args.deploy_cfg_path
    model_cfg_path = args.model_cfg_path
    quant_table_path = args.quant_table_path

    logger.info(f'onnx2ncnn: \n\tonnx_path: {onnx_path} ')
    try:
        gen_quant_table(onnx_path, deploy_cfg_path, model_cfg_path,
                        quant_table_path, 'quantized.onnx')
        logger.info('onnx2ncnn success.')
    except Exception as e:
        logger.error(e)
        logger.error('onnx2ncnn failed.')


if __name__ == '__main__':
    main()
