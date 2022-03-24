# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import yaml
from mmcv.parallel import MMDataParallel
from torch.hub import download_url_to_file
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import torch2onnx, build_task_processor
from mmdeploy.utils import get_root_logger, load_config, target_wrapper, parse_device_id
from mmdeploy.utils.timer import TimeCounter


def parse_args():
    parser = argparse.ArgumentParser(description='Process Regression Test')
    parser.add_argument('--deploy-yml', help='regression test yaml path',
                        default='./configs/mmdet/mmdet_regression_test.yaml')
    parser.add_argument('--test-type', help='`test type', default="precision")
    parser.add_argument('--backend', help='test specific backend(s)',
                        default="all")
    parser.add_argument('--work-dir', help='the dir to save logs and models',
                        default='../mmdeploy_regression_working_dir')
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

    Returns:
        Dict: Meta infos of each model config
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
        download_url_to_file(weights_url, str(weights_save_path), progress=True)

        # check weather the weight download successful
        if not weights_save_path.exists():
            raise FileExistsError(f'Weight {weights_name} download fail')

    logger.info(f'All models had been downloaded successful.')
    return model_meta_info, checkpoint_save_dir, codebase_dir


def update_report(report_dict,
                  model_name,
                  model_config,
                  model_checkpoint_name,
                  dataset,
                  backend_name,
                  deploy_config,
                  static_or_dynamic,
                  conversion_result,
                  fps,
                  metric_info,
                  test_pass,
                  ):
    """Update report information

    Args:
        report_dict (dict): Report info dict.
        model_name (str): Model name.
        model_config (str): Model config name.
        model_checkpoint_name (str): Model checkpoint name.
        dataset (str): Dataset name.
        backend_name (str): Backend name.
        deploy_config (str): Deploy config name.
        static_or_dynamic (str): Static or dynamic.
        conversion_result (str): Conversion result: Successful or Fail.
        fps (str): Inference speed (ms/im).
        metric_info (list): Metric info list of the ${modelName}.yml.
        test_pass (str): Test result: Pass or Fail.
    """
    report_dict.get('model_name').append(model_name)
    report_dict.get('model_config').append(model_config)
    report_dict.get('model_checkpoint_name').append(model_checkpoint_name)
    report_dict.get('dataset').append(dataset)
    report_dict.get('backend_name').append(backend_name)
    report_dict.get('deploy_config').append(deploy_config)
    report_dict.get('static_or_dynamic').append(static_or_dynamic)
    report_dict.get('conversion_result').append(conversion_result)
    report_dict.get('fps').append(fps)

    for metric in metric_info:
        for metric_name, metric_value in metric.items():
            metric_name = str(metric_name).replace(' ', '_')
            report_dict.get(metric_name).append(metric_value)

    report_dict.get('test_pass').append(test_pass)


def get_pytorch_result(model_name, meta_info, checkpoint_path, model_config_name, metric_tolerance, report_dict,
                       logger):
    """Get metric from metafile info of the model

    Args:
        model_name (str): Name of model.
        meta_info (dict): Metafile info from model's metafile.yml.
        checkpoint_path (Path): Checkpoint path.
        metric_tolerance (dict):Tolerance for metrics.
        model_config_name (Path):  Model config name for getting meta info
        report_dict (dict): Report info dict.
        logger (logging.Logger): Logger.

    Returns:
        Dict: metric info of the model
    """

    if model_config_name.stem not in meta_info:
        return {}

    model_info = meta_info.get(model_config_name.stem, None)

    # get metric
    metric_info = model_info.get('Results', None)
    metric_list = []
    pytorch_metric = dict()
    for metric in metric_info:
        metric_list.append(metric.get('Metrics'))
        pytorch_metric.update(metric.get('Metrics'))

    # update useless metric
    metric_all_list = [str(metric) for metric in metric_tolerance]
    metric_useless = set(metric_all_list) - set([str(metric).replace(' ', '_') for metric in pytorch_metric])
    for metric in metric_useless:
        metric_list.append({metric: '-'})

    # get pytorch fps value
    fps_info = model_info.get('Metadata').get('inference time (ms/im)')
    if fps_info is None:
        fps = "-"
    elif isinstance(fps_info, list):
        fps = fps_info[0].get('value')
    else:
        fps = fps_info.get('value')

    # update report
    dataset_type = ""
    for metric in metric_info:
        dataset_type += f"{metric.get('Dataset')},"

    update_report(
        report_dict=report_dict,
        model_name=model_name,
        model_config=str(model_config_name),
        model_checkpoint_name=str(checkpoint_path),
        dataset=dataset_type,
        backend_name='Pytorch',
        deploy_config='-',
        static_or_dynamic='-',
        conversion_result='-',
        fps=fps,
        metric_info=metric_list,
        test_pass='-'
    )

    logger.info(f'Got {model_config_name} metric: {pytorch_metric}')
    return pytorch_metric


def get_info_from_log_file(info_type, log_path):
    # get fps from log file
    if log_path.exists():
        with open(log_path, 'r+') as f_log:
            lines = f_log.readlines()
        with open(log_path, 'w') as f_log:
            f_log.truncate()
    else:
        lines = []

    if info_type == 'FPS' and len(lines) > 1:
        line_count = 0
        fps_sum = 0.00
        for line in lines[-6:]:
            if 'FPS' not in line:
                continue
            line_count += 1
            fps_sum += float(line.split(' ')[-2])
        info_value = f'{fps_sum / line_count:.2f}'
    # elif info_type == 'metric' and len(lines) < 2:
    #     for line in lines:
    #         if 'OrderedDict' not in line:
    #             continue
    #     info_value = '0.00'
    else:
        info_value = '0.00'

    return info_value


def test_backends(deploy_cfg, model_cfg, checkpoint_path, device, work_dir,
                  metrics_name, logger):
    """Test the backend.

    Args:
        deploy_cfg (mmcv.Config): Deploy config file path.
        model_cfg (mmcv.Config): Model config file path.
        checkpoint_path (Path): Backend converted model file path.
        device (str): A string specifying device, defaults to 'cuda:0'.
        work_dir (str): A working directory.
        metrics_name (str): Evaluation metrics, which depends on
            the codebase and the dataset, e.g., "bbox", "segm", "proposal"
            for COCO, and "mAP", "recall" for PASCAL VOC in mmdet;
            "accuracy", "precision", "recall", "f1_score", "support"
            for single label dataset, and "mAP", "CP", "CR", "CF1",
            "OP", "OR", "OF1" for multi-label dataset in mmcls.
            Defaults is `None`.
        logger (logging.Logger): Logger.

    Returns:
        Dict: metric info of the model
    """

    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    # prepare the dataset loader
    logger.info('Loading dataset to memory...')
    dataset = task_processor.build_dataset(model_cfg, 'test')
    data_loader = task_processor.build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=model_cfg.data.workers_per_gpu)

    # load the model of the backend
    model = task_processor.init_backend_model([str(checkpoint_path)])

    is_device_cpu = (device == 'cpu')
    device_id = None if is_device_cpu else parse_device_id(device)
    print(f'Got device_id = {device_id}')

    model = MMDataParallel(model, device_ids=[device_id])
    # The whole dataset test wrapped a MMDataParallel class outside the module.
    # As mmcls.apis.test.py single_gpu_test defined, the MMDataParallel needs
    # a 'CLASSES' attribute. So we ensure the MMDataParallel class has the same
    # CLASSES attribute as the inside module.
    if hasattr(model.module, 'CLASSES'):
        model.CLASSES = model.module.CLASSES

    log_path = checkpoint_path.with_name('backend_test_log.log').absolute()
    if log_path.exists():
        with open(log_path, 'w') as f_log:
            f_log.truncate()  # clear the log file
    print(f'Log path = {log_path}')

    result_path = checkpoint_path.with_suffix('.pkl').absolute()
    if result_path.exists():
        result_path.unlink()

    with_sync = not is_device_cpu

    with TimeCounter.activate(
            warmup=10,
            log_interval=100,
            with_sync=with_sync,
            file=str(log_path)):
        outputs = task_processor.single_gpu_test(model, data_loader,
                                                 show=False, out_dir=None)

    fps = get_info_from_log_file('FPS', log_path)
    print(f'Got fps = {fps}')

    # Get metric
    evaluate_result = task_processor.evaluate_outputs(model_cfg,
                                                      outputs,
                                                      dataset,
                                                      metrics=metrics_name,
                                                      out=str(result_path),
                                                      metric_options=None,
                                                      format_only=False,
                                                      log_file=str(log_path))
    metric = evaluate_result.get(metrics_name, '')
    if not isinstance(metric, float):
        metric = f'{metric:.2f}'
    print(f'Got metric = {metric}')

    return metric, fps


def create_process(name, target, args, kwargs, ret_value=None):
    logger = get_root_logger()
    logger.info(f'{name} start.')
    log_level = logger.level

    wrap_func = partial(target_wrapper, target, log_level, ret_value)

    process = Process(target=wrap_func, args=args, kwargs=kwargs)
    process.start()
    process.join()

    if ret_value is not None:
        if ret_value.value != 0:
            logger.error(f'{name} failed.')
            return False
        else:
            logger.info(f'{name} success.')
            return True

    return False


def get_backend_result(backends_info, model_cfg_path, deploy_config_dir, checkpoint_path,
                       work_dir, device, pytorch_metric, metric_tolerance, report_dict,
                       logger, backends_name):
    """Convert model to onnx and then get metric.

    Args:
        backends_info (dict):  Backend info of test yaml.
        model_cfg_path (Path): Model config file path.
        deploy_config_dir (str): Deploy config directory.
        checkpoint_path (Path): Checkpoints path.
        work_dir (Path): A working directory.
        device (str): A string specifying device, defaults to '0'.
        pytorch_metric (dict): All pytorch metric info.
        metric_tolerance (dict):Tolerance for metrics.
        report_dict (dict): Report info dict.
        logger (logging.Logger): Logger.
        backends_name (str):  Backend name.

    Returns:
        Dict: metric info of the model
    """

    backends_info = backends_info.get(backends_name, [])
    if len(backends_info) <= 0:
        return {}

    backend_exchange_info = {
        'onnxruntime': {
            'suffix': '.onnx',
            'function': torch2onnx,
            'function_name': 'torch2onnx',
        }
    }

    metric_exchange_dict = {
        # mmdet
        'bbox': 'box_AP',
        'segm': 'mask_AP',
        'proposal': 'PQ',
    }

    performance_align = backends_info.get('performance_align', False)

    metric_name_list = [str(metric).replace(' ', '_') for metric in pytorch_metric]
    assert len(metric_name_list) > 0

    metric_all_list = [str(metric) for metric in metric_tolerance]
    deploy_cfg_path_list = backends_info.get('deploy_config')
    for infer_type, deploy_cfg_info in deploy_cfg_path_list.items():
        for fp_size, deploy_cfg in deploy_cfg_info.items():
            if deploy_cfg is None:
                continue
            fps = ''
            img = (np.ones((425, 640, 3))).astype(np.uint8)  # fake image
            metric_list = []
            test_pass = True

            # file path
            backend_suffix = backend_exchange_info.get(backends_name).get('suffix')
            backend_output_path = Path(work_dir). \
                joinpath(Path(checkpoint_path).parent.parent.name,
                         Path(checkpoint_path).parent.name,
                         infer_type,
                         Path(checkpoint_path).name).with_suffix(backend_suffix).absolute()
            backend_output_path.parent.mkdir(parents=True, exist_ok=True)
            deploy_cfg_path = Path(deploy_config_dir).joinpath(deploy_cfg).absolute()
            logger.info(f'torch2onnx: \n\tmodel_cfg: {model_cfg_path.absolute()} '
                        f'\n\tdeploy_cfg: {deploy_cfg_path}')

            # convert the model
            ret_value = mp.Value('d', 0, lock=False)
            convert_result = create_process(
                backend_exchange_info.get(backends_name).get('function_name', None),
                target=backend_exchange_info.get(backends_name).get('function', None),
                args=(img,
                      str(work_dir),
                      str(backend_output_path),
                      str(deploy_cfg_path),
                      str(model_cfg_path),
                      str(checkpoint_path)),
                kwargs=dict(device=device),
                ret_value=ret_value)

            # Test the model
            if convert_result and \
                    infer_type == 'dynamic' and \
                    performance_align:

                # load deploy_cfg
                deploy_cfg, model_cfg = load_config(str(deploy_cfg_path),
                                                    str(model_cfg_path.absolute()))
                # Get evaluation metric from model config
                metrics_eval_list = model_cfg.evaluation.get('metric', [])
                assert len(metrics_eval_list) > 0
                print(f'Got metrics_eval_list = {metrics_eval_list}')

                # test the model metric
                for metric_name in metrics_eval_list:
                    metric_value, fps = test_backends(deploy_cfg=deploy_cfg,
                                                      model_cfg=model_cfg,
                                                      checkpoint_path=backend_output_path.resolve(),
                                                      device='cuda' if device != '-1' else 'cpu',
                                                      work_dir=str(backend_output_path.parent),
                                                      metrics_name=metric_name,
                                                      logger=logger)

                    metric_name = metric_exchange_dict.get(metric_name, None)
                    if metric_name is None:
                        print(f'metrics_eval_list: {metrics_eval_list} has not exchange name')
                    assert metric_name is not None

                    metric_list.append({metric_name: metric_value})
                    metric_pytorch = pytorch_metric.get(str(metric_name).replace('_', ' '))
                    metric_tolerance_value = metric_tolerance.get(metric_name)
                    if (metric_value - metric_tolerance_value) <= \
                            metric_pytorch < \
                            (metric_value + metric_tolerance_value):
                        test_pass = True
                    else:
                        test_pass = False
            else:
                for metric in metric_name_list:
                    metric_list.append({metric: '-'})
                test_pass = False

            # update useless metric
            metric_useless = set(metric_all_list) - set(metric_name_list)
            for metric in metric_useless:
                metric_list.append({metric: '-'})

            # update the report
            update_report(
                report_dict=report_dict,
                model_name=model_cfg_path.parent.name,
                model_config=str(model_cfg_path),
                model_checkpoint_name=str(checkpoint_path),
                dataset='',
                backend_name='onnxruntime',
                deploy_config=str(deploy_cfg_path),
                static_or_dynamic=infer_type,
                conversion_result=str(convert_result),
                fps=fps,
                metric_info=metric_list,
                test_pass=str(test_pass)
            )


def save_report(report_info, report_save_path, logger):
    """Convert model to onnx and then get metric.

    Args:
        report_info (dict):  Report info dict.
        report_save_path (Path): Report save path.
        logger (logging.Logger): Logger.
    """
    logger.info(f'Save regression test report '
                f'to {report_save_path}, pls wait...')

    df = pd.DataFrame(report_info)
    df.to_excel(report_save_path)

    logger.info(f'Saved regression test report '
                f'to {report_save_path}.')


def main():
    args = parse_args()

    set_start_method('spawn')

    logger = get_root_logger(log_level=args.log_level)
    logger.info('Processing regression test.')

    deploy_yaml_list = str(args.deploy_yml).replace(' ', '').split(',')
    assert len(deploy_yaml_list) > 0

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    for deploy_yaml in deploy_yaml_list:

        if not Path(deploy_yaml).exists():
            raise FileNotFoundError(f'deploy_yaml {deploy_yaml} not found, '
                                    'please check !')

        with open(deploy_yaml) as f:
            yaml_info = yaml.load(f, Loader=yaml.FullLoader)

        report_save_path = work_dir.joinpath(Path(deploy_yaml).stem + '_report.xlsx')
        report_dict = {
            'model_name': [],
            'model_config': [],
            'model_checkpoint_name': [],
            'dataset': [],
            'backend_name': [],
            'deploy_config': [],
            'static_or_dynamic': [],
            'conversion_result': [],
            'fps': []
        }

        global_info = yaml_info.get('globals')

        for metric_name in global_info.get('metric_tolerance', {}):
            report_dict.update({metric_name: []})
        metric_tolerance = global_info.get('metric_tolerance', {})

        report_dict.update({'test_pass': []})

        models_info = yaml_info.get('models')

        for models in models_info:
            if 'model_configs' not in models:
                continue

            model_metafile_info, checkpoint_save_dir, codebase_dir = \
                get_model_metafile_info(global_info, models, logger)
            for model_config in model_metafile_info:
                logger.info(f'Processing regression test for {model_config}.py...')

                # get backends info
                backends_info = models.get('backends', None)
                if backends_info is None:
                    continue

                # get model config path
                model_cfg_path = Path(codebase_dir). \
                    joinpath(models.get("codebase_model_config_dir", ""), model_config).with_suffix('.py')
                assert model_cfg_path.exists()

                # get checkpoint path
                checkpoint_name = Path(model_metafile_info.get(model_config).get('Weights')).name
                checkpoint_path = Path(checkpoint_save_dir).joinpath(checkpoint_name)
                assert checkpoint_path.exists()

                # get deploy config directory
                deploy_config_dir = models.get('deploy_config_dir', '')
                assert deploy_config_dir != ''

                pytorch_metric = get_pytorch_result(models.get('name'), model_metafile_info, checkpoint_path,
                                                    model_cfg_path, metric_tolerance, report_dict, logger)

                backend_result_function = partial(get_backend_result, backends_info, model_cfg_path, deploy_config_dir,
                                                  checkpoint_path, work_dir, args.device_id, pytorch_metric,
                                                  metric_tolerance, report_dict, logger)

                backend_result_function('onnxruntime')
                backend_result_function('tensorrt')
                backend_result_function('openvino')
                backend_result_function('ncnn')
                backend_result_function('pplnn')
                backend_result_function('sdk')

        save_report(report_dict, report_save_path, logger)


if __name__ == '__main__':
    main()
