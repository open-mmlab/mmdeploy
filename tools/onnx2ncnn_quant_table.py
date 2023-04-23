# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
from copy import deepcopy

from mmengine import Config
from torch.utils.data import DataLoader

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_root_logger, load_config


def get_table(onnx_path: str,
              deploy_cfg: Config,
              model_cfg: Config,
              output_onnx_path: str,
              output_quant_table_path: str,
              image_dir: str = None,
              device: str = 'cuda',
              dataset_type: str = 'val'):

    input_shape = None
    # setup input_shape if existed in `onnx_config`
    if 'onnx_config' in deploy_cfg and 'input_shape' in deploy_cfg.onnx_config:
        input_shape = deploy_cfg.onnx_config.input_shape

    task_processor = build_task_processor(model_cfg, deploy_cfg, device)
    calib_dataloader = deepcopy(model_cfg[f'{dataset_type}_dataloader'])
    calib_dataloader['batch_size'] = 1
    # build calibration dataloader. If img dir not specified, use val dataset.
    if image_dir is not None:
        from quant_image_dataset import QuantizationImageDataset
        dataset = QuantizationImageDataset(
            path=image_dir, deploy_cfg=deploy_cfg, model_cfg=model_cfg)

        def collate(data_batch):
            return data_batch[0]

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    else:
        dataset = task_processor.build_dataset(calib_dataloader['dataset'])
        calib_dataloader['dataset'] = dataset
        dataloader = task_processor.build_dataloader(calib_dataloader)

    data_preprocessor = task_processor.build_data_preprocessor()

    # get an available input shape randomly
    for _, input_data in enumerate(dataloader):
        input_data = data_preprocessor(input_data)
        input_tensor = input_data['inputs']
        input_shape = input_tensor.shape
        collate_fn = lambda x: data_preprocessor(x)['inputs'].to(  # noqa: E731
            device)

    from ppq import QuantizationSettingFactory, TargetPlatform
    from ppq.api import export_ppq_graph, quantize_onnx_model

    # settings for ncnn quantization
    quant_setting = QuantizationSettingFactory.default_setting()
    quant_setting.equalization = False
    quant_setting.dispatcher = 'conservative'

    # quantize the model
    quantized = quantize_onnx_model(
        onnx_import_file=onnx_path,
        calib_dataloader=dataloader,
        calib_steps=max(8, min(512, len(dataset))),
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
    parser = argparse.ArgumentParser(
        description='Generate ncnn quant table from ONNX.')
    parser.add_argument('--onnx', help='ONNX model path')
    parser.add_argument('--deploy-cfg', help='Input deploy config path')
    parser.add_argument('--model-cfg', help='Input model config path')
    parser.add_argument('--out-onnx', help='Output onnx path')
    parser.add_argument('--out-table', help='Output quant table path')
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help='Calibration Image Directory.')
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

    onnx_path = args.onnx
    deploy_cfg, model_cfg = load_config(args.deploy_cfg, args.model_cfg)
    quant_table_path = args.out_table
    quant_onnx_path = args.out_onnx
    image_dir = args.image_dir

    get_table(onnx_path, deploy_cfg, model_cfg, quant_onnx_path,
              quant_table_path, image_dir)
    logger.info('onnx2ncnn_quant_table success.')


if __name__ == '__main__':
    main()
