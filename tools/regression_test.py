# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import subprocess
from collections import OrderedDict
from pathlib import Path

import mmcv
import openpyxl
import pandas as pd
import yaml
from torch.hub import download_url_to_file
from torch.multiprocessing import set_start_method

import mmdeploy.version
from mmdeploy.utils import (get_backend, get_codebase, get_root_logger,
                            is_dynamic_shape, load_config)


def parse_args():
    parser = argparse.ArgumentParser(description='Regression Test')
    parser.add_argument(
        '--codebase',
        nargs='+',
        help='regression test yaml path.',
        default=[
            'mmcls', 'mmdet', 'mmseg', 'mmpose', 'mmocr', 'mmedit', 'mmrotate'
        ])
    parser.add_argument(
        '-p',
        '--performance',
        default=False,
        action='store_true',
        help='test performance if it set')
    parser.add_argument(
        '--backends', nargs='+', help='test specific backend(s)')
    parser.add_argument('--models', nargs='+', help='test specific model(s)')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='the dir to save logs and models',
        default='../mmdeploy_regression_working_dir')
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='the dir to save checkpoint for all model',
        default='../mmdeploy_checkpoints')
    parser.add_argument(
        '--device', type=str, help='Device type, cuda or cpu', default='cuda')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


def merge_report(work_dir: str, logger: logging.Logger):
    """Merge all the report into one report.

    Args:
        work_dir (str): Work dir that including all reports.
        logger (logging.Logger): Logger handler.
    """
    work_dir = Path(work_dir)
    res_file = work_dir.joinpath(
        f'mmdeploy_regression_test_{mmdeploy.version.__version__}.xlsx')
    logger.info(f'Whole result report saving in {res_file}')

    if res_file.exists():
        # delete if it existed
        res_file.unlink()

    for report_file in work_dir.iterdir():
        if '_report.xlsx' not in report_file.name or \
                report_file.name == res_file.name or \
                not report_file.is_file():
            # skip other file
            continue
        # get info from report
        logger.info(f'Merging {report_file}')
        df = pd.read_excel(str(report_file))
        df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)

        # get key then convert to list
        keys = list(df.keys())
        values = df.values.tolist()

        # sheet name
        sheet_name = report_file.stem.split('_')[0]

        # begin to write
        if res_file.exists():
            # load if it existed
            wb = openpyxl.load_workbook(str(res_file))
        else:
            wb = openpyxl.Workbook()

        # delete if sheet already exist
        if sheet_name in wb.sheetnames:
            wb.remove(wb[sheet_name])
        # create sheet
        wb.create_sheet(title=sheet_name, index=0)
        # write in row
        wb[sheet_name].append(keys)
        for value in values:
            wb[sheet_name].append(value)
        # delete the blank sheet
        for name in wb.sheetnames:
            ws = wb[name]
            if ws.cell(1, 1).value is None:
                wb.remove(ws)
        # save to file
        wb.save(str(res_file))

    logger.info('Report merge successful.')


def get_model_metafile_info(global_info: dict, model_info: dict,
                            logger: logging.Logger):
    """Get model metafile information.

    Args:
        global_info (dict): global info from deploy yaml.
        model_info (dict):  model info from deploy yaml.
        logger (logging.Logger): Logger handler.

    Returns:
        Dict: Meta info of each model config
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
    model_name = _filter_string(model_info.get('name', 'model'))
    checkpoint_save_dir = Path(checkpoint_dir).joinpath(
        codebase_name, model_name)
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving checkpoint in {checkpoint_save_dir}')

    # get model metafile info
    metafile_path = Path(codebase_dir).joinpath(model_info.get('metafile'))
    with open(metafile_path) as f:
        metafile_info = yaml.load(f, Loader=yaml.FullLoader)

    model_meta_info = dict()
    for meta_model in metafile_info.get('Models'):
        if str(meta_model.get('Config')) not in model_config_files:
            # skip if the model not in model_config_files
            logger.warning(f'{meta_model.get("Config")} '
                           f'not in {model_config_files}, pls check ! '
                           'Skip it...')
            continue

        # get meta info
        model_meta_info.update({meta_model.get('Config'): meta_model})

        # get weight url
        weights_url = meta_model.get('Weights')
        weights_name = str(weights_url).split('/')[-1]
        weights_save_path = checkpoint_save_dir.joinpath(weights_name)
        if weights_save_path.exists() and \
                not global_info.get('checkpoint_force_download', False):
            logger.info(f'model {weights_name} exist, skip download it...')
            continue

        # Download weight
        logger.info(f'Downloading {weights_url} to {weights_save_path}')
        download_url_to_file(
            weights_url, str(weights_save_path), progress=True)

        # check weather the weight download successful
        if not weights_save_path.exists():
            raise FileExistsError(f'Weight {weights_name} download fail')

    logger.info('All models had been downloaded successful !')
    return model_meta_info, checkpoint_save_dir, codebase_dir


def update_report(report_dict: dict, model_name: str, model_config: str,
                  task_name: str, checkpoint: str, dataset: str,
                  backend_name: str, deploy_config: str,
                  static_or_dynamic: str, precision_type: str,
                  conversion_result: str, fps: str, metric_info: list,
                  test_pass: str, report_txt_path: Path, codebase_name: str):
    """Update report information.

    Args:
        report_dict (dict): Report info dict.
        model_name (str): Model name.
        model_config (str): Model config name.
        task_name (str): Task name.
        checkpoint (str): Model checkpoint name.
        dataset (str): Dataset name.
        backend_name (str): Backend name.
        deploy_config (str): Deploy config name.
        static_or_dynamic (str): Static or dynamic.
        precision_type (str): Precision type of the model.
        conversion_result (str): Conversion result: Successful or Fail.
        fps (str): Inference speed (ms/im).
        metric_info (list): Metric info list of the ${modelName}.yml.
        test_pass (str): Test result: Pass or Fail.
        report_txt_path (Path): Report txt path.
        codebase_name (str): Codebase name.
    """
    # make model path shorter
    if '.pth' in checkpoint:
        checkpoint = Path(checkpoint).absolute().resolve()
        checkpoint = str(checkpoint).split(f'/{codebase_name}/')[-1]
        checkpoint = '${CHECKPOINT_DIR}' + f'/{codebase_name}/{checkpoint}'
    else:
        if Path(checkpoint).exists():
            # To invoice the path which is 'A.a B.b' when test sdk.
            checkpoint = Path(checkpoint).absolute().resolve()
        elif backend_name == 'ncnn':
            # ncnn have 2 backend file but only need xxx.param
            checkpoint = checkpoint.split('.param')[0] + '.param'
        work_dir = report_txt_path.parent.absolute().resolve()
        checkpoint = str(checkpoint).replace(str(work_dir), '${WORK_DIR}')

    # save to tmp file
    tmp_str = f'{model_name},{model_config},{task_name},{checkpoint},' \
              f'{dataset},{backend_name},{deploy_config},' \
              f'{static_or_dynamic},{precision_type},{conversion_result},' \
              f'{fps},'

    # save to report
    report_dict.get('Model').append(model_name)
    report_dict.get('Model Config').append(model_config)
    report_dict.get('Task').append(task_name)
    report_dict.get('Checkpoint').append(checkpoint)
    report_dict.get('Dataset').append(dataset)
    report_dict.get('Backend').append(backend_name)
    report_dict.get('Deploy Config').append(deploy_config)
    report_dict.get('Static or Dynamic').append(static_or_dynamic)
    report_dict.get('Precision Type').append(precision_type)
    report_dict.get('Conversion Result').append(conversion_result)
    # report_dict.get('FPS').append(fps)

    for metric in metric_info:
        for metric_name, metric_value in metric.items():
            metric_name = str(metric_name)
            report_dict.get(metric_name).append(metric_value)
            tmp_str += f'{metric_value},'
    report_dict.get('Test Pass').append(test_pass)

    tmp_str += f'{test_pass}\n'

    with open(report_txt_path, 'a+', encoding='utf-8') as f:
        f.write(tmp_str)


def get_pytorch_result(model_name: str, meta_info: dict, checkpoint_path: Path,
                       model_config_path: Path, model_config_name: str,
                       test_yaml_metric_info: dict, report_dict: dict,
                       logger: logging.Logger, report_txt_path: Path,
                       codebase_name: str):
    """Get metric from metafile info of the model.

    Args:
        model_name (str): Name of model.
        meta_info (dict): Metafile info from model's metafile.yml.
        checkpoint_path (Path): Checkpoint path.
        model_config_path (Path): Model config path.
        model_config_name (str): Name of model config in meta_info.
        test_yaml_metric_info (dict): Metrics info from test yaml.
        report_dict (dict): Report info dict.
        logger (logging.Logger): Logger.
        report_txt_path (Path): Report txt path.
        codebase_name (str): Codebase name.

    Returns:
        Dict: metric info of the model
    """

    if model_config_name not in meta_info:
        logger.warning(
            f'{model_config_name} not in meta_info, which is {meta_info}')
        return {}

    # get metric
    model_info = meta_info.get(model_config_name, None)
    metafile_metric_info = model_info.get('Results', None)

    metric_list = []
    pytorch_metric = dict()
    dataset_type = ''
    task_type = ''

    # Get dataset
    using_dataset = dict()
    for _, v in test_yaml_metric_info.items():
        if v.get('dataset') is None:
            continue
        dataset_list = v.get('dataset', [])
        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]
        for metric_dataset in dataset_list:
            dataset_tmp = using_dataset.get(metric_dataset, [])
            if v.get('task_name') not in dataset_tmp:
                dataset_tmp.append(v.get('task_name'))
            using_dataset.update({metric_dataset: dataset_tmp})

    # Get metrics info from metafile
    for metafile_metric in metafile_metric_info:
        pytorch_meta_metric = metafile_metric.get('Metrics')

        dataset = metafile_metric.get('Dataset', '')
        task_name = metafile_metric.get('Task', '')

        if task_name == 'Restorers':
            # mmedit
            dataset = 'Set5'

        if (len(using_dataset) > 1) and (dataset not in using_dataset):
            logger.info(f'dataset not in {using_dataset}, skip it...')
            continue
        dataset_type += f'{dataset} | '

        if task_name not in using_dataset.get(dataset, []):
            # only add the metric with the correct dataset
            logger.info(f'task_name ({task_name}) is not in'
                        f'{using_dataset.get(dataset, [])}, skip it...')
            continue
        task_type += f'{task_name} | '

        # remove some metric which not in metric_info from test yaml
        for k, v in pytorch_meta_metric.items():

            if k not in test_yaml_metric_info and \
                    'Restorers' not in task_type:
                continue

            if 'Restorers' in task_type and k not in dataset_type:
                # mmedit
                continue

            if isinstance(v, dict):
                # mmedit
                for sub_k, sub_v in v.items():
                    use_metric = {sub_k: sub_v}
                    metric_list.append(use_metric)
                    pytorch_metric.update(use_metric)
            else:
                use_metric = {k: v}
                metric_list.append(use_metric)
                pytorch_metric.update(use_metric)

    dataset_type = dataset_type[:-3].upper()  # remove the final ' | '
    task_type = task_type[:-3]  # remove the final ' | '

    # update useless metric
    metric_all_list = [str(metric) for metric in test_yaml_metric_info]
    metric_useless = set(metric_all_list) - set(
        [str(metric) for metric in pytorch_metric])
    for metric in metric_useless:
        metric_list.append({metric: '-'})

    # get pytorch fps value
    fps_info = model_info.get('Metadata').get('inference time (ms/im)')
    if fps_info is None:
        fps = '-'
    elif isinstance(fps_info, list):
        fps = fps_info[0].get('value')
    else:
        fps = fps_info.get('value')

    logger.info(f'Got metric_list = {metric_list} ')
    logger.info(f'Got pytorch_metric = {pytorch_metric} ')

    # update report
    update_report(
        report_dict=report_dict,
        model_name=model_name,
        model_config=str(model_config_path),
        task_name=task_type,
        checkpoint=str(checkpoint_path),
        dataset=dataset_type,
        backend_name='Pytorch',
        deploy_config='-',
        static_or_dynamic='-',
        precision_type='-',
        conversion_result='-',
        fps=fps,
        metric_info=metric_list,
        test_pass='-',
        report_txt_path=report_txt_path,
        codebase_name=codebase_name)

    logger.info(f'Got {model_config_path} metric: {pytorch_metric}')
    return pytorch_metric, dataset_type


def get_info_from_log_file(info_type: str, log_path: Path,
                           yaml_metric_key: str, logger: logging.Logger):
    """Get fps and metric result from log file.

    Args:
        info_type (str): Get which type of info: 'FPS' or 'metric'.
        log_path (Path): Logger path.
        yaml_metric_key (str): Name of metric from yaml metric_key.
        logger (logger.Logger): Logger handler.

    Returns:
        Float: Info value which get from logger file.
    """
    if log_path.exists():
        with open(log_path, 'r') as f_log:
            lines = f_log.readlines()
    else:
        logger.warning(f'{log_path} do not exist !!!')
        lines = []

    if info_type == 'FPS' and len(lines) > 1:
        # Get FPS
        line_count = 0
        fps_sum = 0.00
        fps_lines = lines[1:11]

        for line in fps_lines:
            if 'FPS' not in line:
                continue
            line_count += 1
            fps_sum += float(line.split(' ')[-2])
        if fps_sum > 0.00:
            info_value = f'{fps_sum / line_count:.2f}'
        else:
            info_value = 'x'

    elif info_type == 'metric' and len(lines) > 1:
        # To calculate the final line index
        if lines[-1] != '' and lines[-1] != '\n':
            line_index = -1
        else:
            line_index = -2

        if yaml_metric_key in ['accuracy_top-1', 'mIoU', 'Eval-PSNR']:
            # info in last second line
            # mmcls, mmseg, mmedit
            metric_line = lines[line_index - 1]
        elif yaml_metric_key == 'AP':
            # info in last tenth line
            # mmpose
            metric_line = lines[line_index - 9]
        elif yaml_metric_key == 'AR':
            # info in last fifth line
            # mmpose
            metric_line = lines[line_index - 4]
        else:
            # info in final line
            # mmdet
            metric_line = lines[line_index]
        logger.info(f'Got metric_line = {metric_line}')

        metric_str = \
            metric_line.replace('\n', '').replace('\r', '').split(' - ')[-1]
        logger.info(f'Got metric_str = {metric_str}')
        logger.info(f'Got metric_info = {yaml_metric_key}')

        if 'OrderedDict' in metric_str:
            # mmdet
            evaluate_result = eval(metric_str)
            if not isinstance(evaluate_result, OrderedDict):
                logger.warning(f'Got error metric_dict = {metric_str}')
                return 'x'
            metric = evaluate_result.get(yaml_metric_key, 0.00) * 100
        elif 'accuracy_top' in metric_str:
            # mmcls
            metric = eval(metric_str.split(': ')[-1])
            if metric <= 1:
                metric *= 100
        elif yaml_metric_key == 'mIoU' and '|' in metric_str:
            # mmseg
            metric = eval(metric_str.strip().split('|')[2])
            if metric <= 1:
                metric *= 100
        elif yaml_metric_key in ['AP', 'AR']:
            # mmpose
            metric = eval(metric_str.split(': ')[-1])
        elif yaml_metric_key == '0_word_acc_ignore_case' or \
                yaml_metric_key == '0_hmean-iou:hmean':
            # mmocr
            evaluate_result = eval(metric_str)
            if not isinstance(evaluate_result, dict):
                logger.warning(f'Got error metric_dict = {metric_str}')
                return 'x'
            metric = evaluate_result.get(yaml_metric_key, 0.00)
            if yaml_metric_key == '0_word_acc_ignore_case':
                metric *= 100
        elif yaml_metric_key in ['Eval-PSNR', 'Eval-SSIM']:
            # mmedit
            metric = eval(metric_str.split(': ')[-1])
        else:
            metric = 'x'
        info_value = metric
    else:
        info_value = 'x'

    return info_value


def compare_metric(metric_value: float, metric_name: str, pytorch_metric: dict,
                   metric_info: dict):
    """Compare metric value with the pytorch metric value and the tolerance.

    Args:
        metric_value (float): Metric value.
        metric_name (str): metric name.
        pytorch_metric (dict): Pytorch metric which get from metafile.
        metric_info (dict): Metric info from test yaml.

    Returns:
        Bool: If the test pass or not.
    """
    if metric_value == 'x':
        return False

    metric_pytorch = pytorch_metric.get(str(metric_name))
    tolerance_value = metric_info.get(metric_name, {}).get('tolerance', 0.00)
    if (metric_value - tolerance_value) <= metric_pytorch <= \
            (metric_value + tolerance_value):
        test_pass = True
    else:
        test_pass = False
    return test_pass


def get_fps_metric(shell_res: int, pytorch_metric: dict, metric_key: str,
                   yaml_metric_info_name: str, log_path: Path,
                   metrics_eval_list: dict, metric_info: dict,
                   logger: logging.Logger):
    """Get fps and metric.

    Args:
        shell_res (int): Backend convert result: 0 is success.
        pytorch_metric (dict): Metric info of pytorch metafile.
        metric_key (str):Metric info.
        yaml_metric_info_name (str): Name of metric info in test yaml.
        log_path (Path): Logger path.
        metrics_eval_list (dict): Metric list from test yaml.
        metric_info (dict): Metric info.
        logger (logger.Logger): Logger handler.

    Returns:
        Float: fps: FPS of the model.
        List: metric_list: metric result list.
        Bool: test_pass: If the test pass or not.
    """
    metric_list = []

    # check if converted successes or not.
    if shell_res != 0:
        fps = 'x'
        metric_value = 'x'
    else:
        # Got fps from log file
        fps = get_info_from_log_file('FPS', log_path, metric_key, logger)
        # logger.info(f'Got fps = {fps}')

        # Got metric from log file
        metric_value = get_info_from_log_file('metric', log_path, metric_key,
                                              logger)
        logger.info(f'Got metric = {metric_value}')

    if yaml_metric_info_name is None:
        logger.error(f'metrics_eval_list: {metrics_eval_list} '
                     'has not metric name')
    assert yaml_metric_info_name is not None

    metric_list.append({yaml_metric_info_name: metric_value})
    test_pass = compare_metric(metric_value, yaml_metric_info_name,
                               pytorch_metric, metric_info)

    # same eval_name and multi metric output in one test
    if yaml_metric_info_name == 'Top 1 Accuracy':
        # mmcls
        yaml_metric_info_name = 'Top 5 Accuracy'
        second_get_metric = True
    elif yaml_metric_info_name == 'AP':
        # mmpose
        yaml_metric_info_name = 'AR'
        second_get_metric = True
    elif yaml_metric_info_name == 'PSNR':
        # mmedit
        yaml_metric_info_name = 'SSIM'
        second_get_metric = True
    else:
        second_get_metric = False

    if second_get_metric:
        metric_key = metric_info.get(yaml_metric_info_name).get('metric_key')
        if shell_res != 0:
            metric_value = 'x'
        else:
            metric_value = get_info_from_log_file('metric', log_path,
                                                  metric_key, logger)
        metric_list.append({yaml_metric_info_name: metric_value})
        if test_pass:
            test_pass = compare_metric(metric_value, yaml_metric_info_name,
                                       pytorch_metric, metric_info)

    return fps, metric_list, test_pass


def get_backend_fps_metric(deploy_cfg_path: str, model_cfg_path: Path,
                           convert_checkpoint_path: str, device_type: str,
                           eval_name: str, logger: logging.Logger,
                           metrics_eval_list: dict, pytorch_metric: dict,
                           metric_info: dict, backend_name: str,
                           precision_type: str, metric_useless: set,
                           convert_result: bool, report_dict: dict,
                           infer_type: str, log_path: Path, dataset_type: str,
                           report_txt_path: Path, model_name: str):
    """Get backend fps and metric.

    Args:
        deploy_cfg_path (str): Deploy config path.
        model_cfg_path (Path): Model config path.
        convert_checkpoint_path (str): Converted checkpoint path.
        device_type (str): Device for converting.
        eval_name (str): Name of evaluation.
        logger (logging.Logger): Logger handler.
        metrics_eval_list (dict): Evaluation metric info dict.
        pytorch_metric (dict): Pytorch metric info dict get from metafile.
        metric_info (dict): Metric info from test yaml.
        backend_name (str): Backend name.
        precision_type (str): Precision type for evaluation.
        metric_useless (set): Useless metric for specific the model.
        convert_result (bool): Backend convert result.
        report_dict (dict): Backend convert result.
        infer_type (str): Infer type.
        log_path (Path): Logger save path.
        dataset_type (str): Dataset type.
        report_txt_path (Path): report txt save path.
        model_name (str): Name of model in test yaml.
    """
    cmd_str = 'python3 tools/test.py ' \
              f'{deploy_cfg_path} ' \
              f'{str(model_cfg_path.absolute())} ' \
              f'--model {convert_checkpoint_path} ' \
              f'--log2file "{log_path}" ' \
              f'--speed-test ' \
              f'--device {device_type} '

    codebase_name = get_codebase(str(deploy_cfg_path)).value
    if codebase_name != 'mmedit':
        # mmedit dont --metric
        cmd_str += f'--metrics {eval_name} '

    logger.info(f'Process cmd = {cmd_str}')

    # Test backend
    shell_res = subprocess.run(
        cmd_str, cwd=str(Path(__file__).absolute().parent.parent),
        shell=True).returncode
    logger.info(f'Got shell_res = {shell_res}')

    metric_key = ''
    metric_name = ''
    task_name = ''
    for key, value in metric_info.items():
        if value.get('eval_name', '') == eval_name:
            metric_name = key
            metric_key = value.get('metric_key', '')
            task_name = value.get('task_name', '')
            break

    logger.info(f'Got metric_name = {metric_name}')
    logger.info(f'Got metric_key = {metric_key}')

    fps, metric_list, test_pass = \
        get_fps_metric(shell_res, pytorch_metric, metric_key, metric_name,
                       log_path, metrics_eval_list, metric_info, logger)

    # update useless metric
    for metric in metric_useless:
        metric_list.append({metric: '-'})

    if len(metrics_eval_list) > 1 and codebase_name == 'mmdet':
        # one model has more than one task, like Mask R-CNN
        for name in pytorch_metric:
            if name in metric_useless or name == metric_name:
                continue
            metric_list.append({name: '-'})

    # update the report
    update_report(
        report_dict=report_dict,
        model_name=model_name,
        model_config=str(model_cfg_path),
        task_name=task_name,
        checkpoint=convert_checkpoint_path,
        dataset=dataset_type,
        backend_name=backend_name,
        deploy_config=str(deploy_cfg_path),
        static_or_dynamic=infer_type,
        precision_type=precision_type,
        conversion_result=str(convert_result),
        fps=fps,
        metric_info=metric_list,
        test_pass=str(test_pass),
        report_txt_path=report_txt_path,
        codebase_name=codebase_name)


def get_precision_type(deploy_cfg_name: str):
    """Get backend precision_type according to the name of deploy config.

    Args:
        deploy_cfg_name (str): Name of the deploy config.

    Returns:
        Str: precision_type: Precision type of the deployment name.
    """
    if 'int8' in deploy_cfg_name:
        precision_type = 'int8'
    elif 'fp16' in deploy_cfg_name:
        precision_type = 'fp16'
    else:
        precision_type = 'fp32'

    return precision_type


def replace_top_in_pipeline_json(backend_output_path: Path,
                                 logger: logging.Logger):
    """Replace `topk` with `num_classes` in `pipeline.json`.

    Args:
        backend_output_path (Path): Backend convert result path.
        logger (logger.Logger): Logger handler.
    """

    sdk_pipeline_json_path = backend_output_path.joinpath('pipeline.json')
    sdk_pipeline_json = mmcv.load(sdk_pipeline_json_path)

    pipeline_tasks = sdk_pipeline_json.get('pipeline', {}).get('tasks', [])
    for index, task in enumerate(pipeline_tasks):
        if task.get('name', '') != 'postprocess':
            continue
        num_classes = task.get('params', {}).get('num_classes', 0)
        if 'topk' not in task.get('params', {}):
            continue
        sdk_pipeline_json['pipeline']['tasks'][index]['params']['topk'] = \
            num_classes

    logger.info(f'sdk_pipeline_json = {sdk_pipeline_json}')

    mmcv.dump(
        sdk_pipeline_json, sdk_pipeline_json_path, sort_keys=False, indent=4)

    logger.info('replace done')


def gen_log_path(backend_output_path: Path, log_name: str):
    log_path = backend_output_path.joinpath(log_name).absolute().resolve()
    if log_path.exists():
        # clear the log file
        with open(log_path, 'w') as f_log:
            f_log.write('')

    return log_path


def get_backend_result(pipeline_info: dict, model_cfg_path: Path,
                       checkpoint_path: Path, work_dir: Path, device_type: str,
                       pytorch_metric: dict, metric_info: dict,
                       report_dict: dict, test_type: str,
                       logger: logging.Logger, backend_file_name: [str, list],
                       report_txt_path: Path, metafile_dataset: str,
                       model_name: str):
    """Convert model to onnx and then get metric.

    Args:
        pipeline_info (dict):  Pipeline info of test yaml.
        model_cfg_path (Path): Model config file path.
        checkpoint_path (Path): Checkpoints path.
        work_dir (Path): A working directory.
        device_type (str): A string specifying device, defaults to 'cuda'.
        pytorch_metric (dict): All pytorch metric info.
        metric_info (dict): Metrics info.
        report_dict (dict): Report info dict.
        test_type (str): Test type. 'precision' or 'convert'.
        logger (logging.Logger): Logger.
        backend_file_name (str | list): backend file save name.
        report_txt_path (Path): report txt path.
        metafile_dataset (str): Dataset type get from metafile.
        model_name (str): Name of model in test yaml.
    """
    # get backend_test info
    backend_test = pipeline_info.get('backend_test', False)

    # get convert_image info
    convert_image_info = pipeline_info.get('convert_image', None)
    if convert_image_info is not None:
        input_img_path = \
            convert_image_info.get('input_img', './tests/data/tiger.jpeg')
        test_img_path = convert_image_info.get('test_img', None)
    else:
        input_img_path = './tests/data/tiger.jpeg'
        test_img_path = None

    # get sdk_cfg info
    sdk_config = pipeline_info.get('sdk_config', None)
    if sdk_config is not None:
        sdk_config = Path(sdk_config)

    # Overwrite metric tolerance
    metric_tolerance = pipeline_info.get('metric_tolerance', None)
    if metric_tolerance is not None:
        for metric, new_tolerance in metric_tolerance.items():
            if metric not in metric_info:
                logger.debug(f'{metric} not in {metric_info},'
                             'skip compare it...')
                continue
            if new_tolerance is None:
                logger.debug('new_tolerance is None, skip it ...')
                continue
            metric_info[metric]['tolerance'] = new_tolerance

    if backend_test is False and sdk_config is None:
        test_type = 'convert'

    metric_name_list = [str(metric) for metric in pytorch_metric]
    assert len(metric_name_list) > 0
    metric_all_list = [str(metric) for metric in metric_info]
    metric_useless = set(metric_all_list) - set(metric_name_list)

    deploy_cfg_path = Path(pipeline_info.get('deploy_config'))
    backend_name = str(get_backend(str(deploy_cfg_path)).name).lower()

    # change device_type for special case
    if backend_name in ['ncnn', 'openvino']:
        device_type = 'cpu'
    elif backend_name == 'onnxruntime' and device_type != 'cpu':
        import onnxruntime as ort
        if ort.get_device() != 'GPU':
            device_type = 'cpu'
            logger.warning('Device type is forced to cpu '
                           'since onnxruntime-gpu is not installed')

    infer_type = \
        'dynamic' if is_dynamic_shape(str(deploy_cfg_path)) else 'static'

    precision_type = get_precision_type(deploy_cfg_path.name)
    codebase_name = get_codebase(str(deploy_cfg_path)).value

    backend_output_path = Path(work_dir). \
        joinpath(Path(checkpoint_path).parent.parent.name,
                 Path(checkpoint_path).parent.name,
                 backend_name,
                 infer_type,
                 precision_type,
                 Path(checkpoint_path).stem)
    backend_output_path.mkdir(parents=True, exist_ok=True)

    # convert cmd string
    cmd_str = 'python3 ./tools/deploy.py ' \
              f'{str(deploy_cfg_path.absolute().resolve())} ' \
              f'{str(model_cfg_path.absolute().resolve())} ' \
              f'"{str(checkpoint_path.absolute().resolve())}" ' \
              f'"{input_img_path}" ' \
              f'--work-dir "{backend_output_path}" ' \
              f'--device {device_type} ' \
              '--log-level INFO'

    if sdk_config is not None:
        cmd_str += ' --dump-info'

    if test_img_path is not None:
        cmd_str += f' --test-img {test_img_path}'

    if precision_type == 'int8':
        calib_dataset_cfg = pipeline_info.get('calib_dataset_cfg', None)
        if calib_dataset_cfg is not None:
            cmd_str += f' --calib-dataset-cfg {calib_dataset_cfg}'

    logger.info(f'Process cmd = {cmd_str}')

    convert_result = False
    convert_log_path = backend_output_path.joinpath('convert_log.log')
    logger.info(f'Logging conversion log to {convert_log_path} ...')
    file_handler = open(convert_log_path, 'w', encoding='utf-8')
    try:
        # Convert the model to specific backend
        process_res = subprocess.Popen(
            cmd_str,
            cwd=str(Path(__file__).absolute().parent.parent),
            shell=True,
            stdout=file_handler,
            stderr=file_handler)
        process_res.wait()
        logger.info(f'Got shell_res = {process_res.returncode}')

        # check if converted successes or not.
        if process_res.returncode == 0:
            convert_result = True
        else:
            convert_result = False

    except Exception as e:
        print(f'process convert error: {e}')
    finally:
        file_handler.close()

    logger.info(f'Got convert_result = {convert_result}')

    if isinstance(backend_file_name, list):
        convert_checkpoint_path = ''
        for backend_file in backend_file_name:
            backend_path = backend_output_path.joinpath(backend_file)
            backend_path = str(backend_path.absolute().resolve())
            convert_checkpoint_path += f'{str(backend_path)} '
    else:
        convert_checkpoint_path = \
            str(backend_output_path.joinpath(backend_file_name))

    # load deploy_cfg
    deploy_cfg, model_cfg = \
        load_config(str(deploy_cfg_path),
                    str(model_cfg_path.absolute()))
    # get dataset type
    if 'dataset_type' in model_cfg:
        dataset_type = \
            str(model_cfg.dataset_type).upper().replace('DATASET', '')
    else:
        dataset_type = metafile_dataset

    # Test the model
    if convert_result and test_type == 'precision':
        # Get evaluation metric from model config
        metrics_eval_list = model_cfg.evaluation.get('metric', [])
        if isinstance(metrics_eval_list, str):
            # some config is using str only
            metrics_eval_list = [metrics_eval_list]

        # assert len(metrics_eval_list) > 0
        logger.info(f'Got metrics_eval_list = {metrics_eval_list}')
        if len(metrics_eval_list) == 0 and codebase_name == 'mmedit':
            metrics_eval_list = ['PSNR']

        # test the model metric
        for metric_name in metrics_eval_list:
            if backend_test:
                log_path = \
                    gen_log_path(backend_output_path, 'backend_test.log')
                get_backend_fps_metric(
                    deploy_cfg_path=str(deploy_cfg_path),
                    model_cfg_path=model_cfg_path,
                    convert_checkpoint_path=convert_checkpoint_path,
                    device_type=device_type,
                    eval_name=metric_name,
                    logger=logger,
                    metrics_eval_list=metrics_eval_list,
                    pytorch_metric=pytorch_metric,
                    metric_info=metric_info,
                    backend_name=backend_name,
                    precision_type=precision_type,
                    metric_useless=metric_useless,
                    convert_result=convert_result,
                    report_dict=report_dict,
                    infer_type=infer_type,
                    log_path=log_path,
                    dataset_type=dataset_type,
                    report_txt_path=report_txt_path,
                    model_name=model_name)

            if sdk_config is not None:

                if codebase_name == 'mmcls':
                    replace_top_in_pipeline_json(backend_output_path, logger)

                log_path = gen_log_path(backend_output_path, 'sdk_test.log')
                # sdk test
                get_backend_fps_metric(
                    deploy_cfg_path=str(sdk_config),
                    model_cfg_path=model_cfg_path,
                    convert_checkpoint_path=str(backend_output_path),
                    device_type=device_type,
                    eval_name=metric_name,
                    logger=logger,
                    metrics_eval_list=metrics_eval_list,
                    pytorch_metric=pytorch_metric,
                    metric_info=metric_info,
                    backend_name=f'SDK-{backend_name}',
                    precision_type=precision_type,
                    metric_useless=metric_useless,
                    convert_result=convert_result,
                    report_dict=report_dict,
                    infer_type=infer_type,
                    log_path=log_path,
                    dataset_type=dataset_type,
                    report_txt_path=report_txt_path,
                    model_name=model_name)
    else:
        logger.info('Only test convert, saving to report...')
        metric_list = []
        fps = '-'

        task_name = ''
        for metric in metric_name_list:
            metric_list.append({metric: '-'})
            metric_task_name = metric_info.get(metric, {}).get('task_name', '')
            if metric_task_name in task_name:
                logger.debug('metric_task_name exist, skip for adding it...')
                continue
            task_name += f'{metric_task_name} | '
        if ' | ' == task_name[-3:]:
            task_name = task_name[:-3]
        test_pass = True if convert_result else False

        # update useless metric
        for metric in metric_useless:
            metric_list.append({metric: '-'})

        if convert_result:
            report_checkpoint = convert_checkpoint_path
        else:
            report_checkpoint = str(checkpoint_path)

        # update the report
        update_report(
            report_dict=report_dict,
            model_name=model_name,
            model_config=str(model_cfg_path),
            task_name=task_name,
            checkpoint=report_checkpoint,
            dataset=dataset_type,
            backend_name=backend_name,
            deploy_config=str(deploy_cfg_path),
            static_or_dynamic=infer_type,
            precision_type=precision_type,
            conversion_result=str(convert_result),
            fps=fps,
            metric_info=metric_list,
            test_pass=str(test_pass),
            report_txt_path=report_txt_path,
            codebase_name=codebase_name)


def save_report(report_info: dict, report_save_path: Path,
                logger: logging.Logger):
    """Convert model to onnx and then get metric.

    Args:
        report_info (dict):  Report info dict.
        report_save_path (Path): Report save path.
        logger (logging.Logger): Logger.
    """
    logger.info('Saving regression test report to '
                f'{report_save_path.absolute().resolve()}, pls wait...')
    try:
        df = pd.DataFrame(report_info)
        df.to_excel(report_save_path)
    except ValueError:
        logger.info(f'Got error report_info = {report_info}')

    logger.info('Saved regression test report to '
                f'{report_save_path.absolute().resolve()}.')


def _filter_string(inputs):
    """Remove non alpha&number character from input string.

    Args:
        inputs(str): Input string.

    Returns:
        str: Output of only alpha&number string.
    """
    outputs = ''.join([i.lower() for i in inputs if i.isalnum()])
    return outputs


def main():
    args = parse_args()
    set_start_method('spawn')
    logger = get_root_logger(log_level=args.log_level)

    test_type = 'precision' if args.performance else 'convert'
    logger.info(f'Processing regression test in {test_type} mode.')

    backend_file_info = {
        'onnxruntime': 'end2end.onnx',
        'tensorrt': 'end2end.engine',
        'openvino': 'end2end.xml',
        'ncnn': ['end2end.param', 'end2end.bin'],
        'pplnn': ['end2end.onnx', 'end2end.json'],
        'torchscript': 'end2end.pt'
    }

    backend_list = args.backends
    if backend_list is None:
        backend_list = [
            'onnxruntime', 'tensorrt', 'openvino', 'ncnn', 'pplnn',
            'torchscript'
        ]
    assert isinstance(backend_list, list)
    logger.info(f'Regression test backend list = {backend_list}')

    if args.models is None:
        logger.info('Regression test for all models in test yaml.')
    else:
        args.models = tuple([_filter_string(s) for s in args.models])
        logger.info(f'Regression test models list = {args.models}')

    assert ' ' not in args.work_dir, \
        f'No empty space included in {args.work_dir}'
    assert ' ' not in args.checkpoint_dir, \
        f'No empty space included in {args.checkpoint_dir}'

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    deploy_yaml_list = [
        f'./tests/regression/{codebase}.yml' for codebase in args.codebase
    ]

    for deploy_yaml in deploy_yaml_list:

        if not Path(deploy_yaml).exists():
            raise FileNotFoundError(f'deploy_yaml {deploy_yaml} not found, '
                                    'please check !')

        with open(deploy_yaml) as f:
            yaml_info = yaml.load(f, Loader=yaml.FullLoader)

        report_save_path = \
            work_dir.joinpath(Path(deploy_yaml).stem + '_report.xlsx')
        report_txt_path = report_save_path.with_suffix('.txt')

        report_dict = {
            'Model': [],
            'Model Config': [],
            'Task': [],
            'Checkpoint': [],
            'Dataset': [],
            'Backend': [],
            'Deploy Config': [],
            'Static or Dynamic': [],
            'Precision Type': [],
            'Conversion Result': [],
            # 'FPS': []
        }

        global_info = yaml_info.get('globals')
        metric_info = global_info.get('metric_info', {})
        for metric_name in metric_info:
            report_dict.update({metric_name: []})
        report_dict.update({'Test Pass': []})

        global_info.update({'checkpoint_dir': args.checkpoint_dir})
        global_info.update(
            {'codebase_name': Path(deploy_yaml).stem.split('_')[0]})

        with open(report_txt_path, 'w') as f_report:
            title_str = ''
            for key in report_dict:
                title_str += f'{key},'
            title_str = title_str[:-1] + '\n'
            f_report.write(title_str)  # clear the report tmp file

        models_info = yaml_info.get('models')
        for models in models_info:
            model_name_origin = models.get('name', 'model')
            model_name_new = _filter_string(model_name_origin)
            if 'model_configs' not in models:
                logger.warning('Can not find field "model_configs", '
                               f'skipping {model_name_origin}...')
                continue

            if args.models is not None and model_name_new not in args.models:
                logger.info(
                    f'Test specific model mode, skip {model_name_origin}...')
                continue

            model_metafile_info, checkpoint_save_dir, codebase_dir = \
                get_model_metafile_info(global_info, models, logger)
            for model_config in model_metafile_info:
                logger.info(f'Processing test for {model_config}...')

                # Get backends info
                pipelines_info = models.get('pipelines', None)
                if pipelines_info is None:
                    logger.warning('pipelines_info is None, skip it...')
                    continue

                # Get model config path
                model_cfg_path = Path(codebase_dir).joinpath(model_config)
                assert model_cfg_path.exists()

                # Get checkpoint path
                checkpoint_name = Path(
                    model_metafile_info.get(model_config).get('Weights')).name

                checkpoint_path = Path(checkpoint_save_dir, checkpoint_name)
                assert checkpoint_path.exists()

                # Get pytorch from metafile.yml
                pytorch_metric, metafile_dataset = get_pytorch_result(
                    model_name_origin, model_metafile_info, checkpoint_path,
                    model_cfg_path, model_config, metric_info, report_dict,
                    logger, report_txt_path, global_info.get('codebase_name'))

                for pipeline in pipelines_info:
                    deploy_config = pipeline.get('deploy_config')
                    backend_name = get_backend(deploy_config).name.lower()
                    if backend_name not in backend_list:
                        logger.warning(f'backend_name ({backend_name}) not '
                                       f'in {backend_list}, skip it...')
                        continue

                    backend_file_name = \
                        backend_file_info.get(backend_name, None)
                    if backend_file_name is None:
                        logger.warning('backend_file_name is None, '
                                       'skip it...')
                        continue

                    get_backend_result(pipeline, model_cfg_path,
                                       checkpoint_path, work_dir, args.device,
                                       pytorch_metric, metric_info,
                                       report_dict, test_type, logger,
                                       backend_file_name, report_txt_path,
                                       metafile_dataset, model_name_origin)
        if len(report_dict.get('Model')) > 0:
            save_report(report_dict, report_save_path, logger)
        else:
            logger.info(f'No model for {deploy_yaml}, not saving report.')

    # merge report
    merge_report(str(work_dir), logger)

    logger.info('All done.')


if __name__ == '__main__':
    main()
