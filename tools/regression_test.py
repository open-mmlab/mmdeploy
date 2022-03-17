# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
from pathlib import Path

import yaml
import torch
from mmdeploy.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Process Regression Test')
    parser.add_argument('--deploy-yml', help='regression test yaml path',
                        default='../configs/mmdet/mmdet_regression_test.yaml')
    parser.add_argument('--test-type', help='`test type', default="precision")
    parser.add_argument('--backend', help='test specific backend(s)')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--device-id', help='`the CUDA device id', default=0)
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


def get_model_metafile_info(global_info, model_info, logger):
    """Get model metafile information.

    Args:
        global_info (dict): global info from deploy yaml.
        model_info (dict):  model info from deploy yaml.
        logger (logging.Logger): logger.
    """

    # get info from global_info and model_info
    checkpoint_dir = global_info.get('checkpoint_dir', None)
    assert checkpoint_dir is not None

    codebase_dir = global_info.get('codebase_dir', None)
    assert codebase_dir is not None

    codebase_name = global_info.get('codebase_name', None)
    assert codebase_name is not None

    model_config_files = model_info.get('model_configs', [])
    assert len(model_config_files) > 0

    # make checkpoint save directory
    checkpoint_save_dir = Path(checkpoint_dir).joinpath(codebase_name,
                                                        model_info.get('name'))
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)

    # get model metafile info
    metafile_path = Path(codebase_dir).joinpath(model_info.get('metafile'))
    with open(metafile_path) as f:
        metafile_info = yaml.load(f, Loader=yaml.FullLoader)

    model_meta_info = dict()
    for meta_model in metafile_info.get('Models'):
        if str(meta_model.get('Name')) + ".py" not in model_config_files:
            # skip if the model not in model_config_files
            continue

        # get meta info
        model_meta_info.update({meta_model.get('Name'): meta_model})

        # get weight url
        weights_url = meta_model.get('Weights')
        weights_name = str(weights_url).split('/')[-1]
        weights_save_path = checkpoint_save_dir.joinpath(weights_name)
        if weights_save_path.exists() and \
                not global_info.get('checkpoint_force_download', False):
            continue

        # Download weight
        logger.info(f'Downloading {weights_url} to {weights_save_path}')
        torch.hub.download_url_to_file(weights_url, str(weights_save_path), progress=True)

        # check weather the weight download successful
        if not weights_save_path.exists():
            raise FileExistsError(f'Weight {weights_name} download fail')

    logger.info(f'All models had been downloaded successful.')
    return model_meta_info


def get_pytorch_result(global_info, model_info, logger):
    """Using pytorch to run the metric

    """
    return False


def convert_model(global_info, model_info, logger):
    return False


def get_onnxruntime_result(global_info, model_info, logger):
    return False


def get_tensorrt_result(global_info, model_info, logger):
    return False


def get_openvino_result(global_info, model_info, logger):
    return False


def get_ncnn_result(global_info, model_info, logger):
    return False


def get_pplnn_result(global_info, model_info, logger):
    return False


def get_sdk_result(global_info, model_info, logger):
    return False


def save_report():
    pass


def main():
    args = parse_args()
    logger = get_root_logger(log_level=args.log_level)

    logger.info('Processing regression test.')

    deploy_yaml_list = str(args.deploy_yml).replace(' ', '').split(',')
    assert len(deploy_yaml_list) > 0

    for deploy_yaml in deploy_yaml_list:

        if not Path(deploy_yaml).exists():
            raise FileNotFoundError(f'deploy_yaml {deploy_yaml} not found, '
                                    'please check !')

        with open(deploy_yaml) as f:
            yaml_info = yaml.load(f, Loader=yaml.FullLoader)

        global_info = yaml_info.get('globals')
        models_info = yaml_info.get('models')

        for models in models_info:
            if 'model_configs' not in models:
                continue

            model_metafile_info = get_model_metafile_info(global_info, models, logger)

            for model_config in model_metafile_info:
                pytorch_result = get_pytorch_result(global_info, models, logger)
                convert_result = convert_model(global_info, models, logger)
                onnxruntime_result = get_onnxruntime_result(global_info, models, logger)
                tensorrt_result = get_tensorrt_result(global_info, models, logger)
                openvino_result = get_openvino_result(global_info, models, logger)
                ncnn_result = get_ncnn_result(global_info, models, logger)
                pplnn_result = get_pplnn_result(global_info, models, logger)
                sdk_result = get_sdk_result(global_info, models, logger)
                save_report()


if __name__ == '__main__':
    main()
