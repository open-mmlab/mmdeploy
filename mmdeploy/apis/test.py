import warnings
from typing import Optional

import mmcv
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mmdeploy.utils import Codebase


def single_gpu_test(codebase: Codebase,
                    model: nn.Module,
                    data_loader: DataLoader,
                    show: bool = False,
                    out_dir: Optional[str] = None,
                    show_score_thr: float = 0.3):
    """Run test with single gpu.

    Args:
        codebase (Codebase): Specifying codebase type.
        model (torch.nn.Module): Input model from nn.Module.
        data_loader (DataLoader): PyTorch data loader.
        show (bool): Specifying whether to show plotted results. Defaults
            to `False`.
        out_dir (str): A directory to save results, defaults to `None`.
        show_score_thr (float): A threshold to show detection results,
            defaults to `0.3`.

    Returns:
        list: The prediction results.
    """
    if codebase == Codebase.MMCLS:
        from mmcls.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir)
    elif codebase == Codebase.MMDET:
        from mmdet.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir,
                                  show_score_thr)
    elif codebase == Codebase.MMSEG:
        from mmseg.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir)
    elif codebase == Codebase.MMOCR:
        from mmdet.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir)
    elif codebase == Codebase.MMEDIT:
        from mmedit.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir)
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')
    return outputs


def post_process_outputs(outputs: list,
                         dataset: Dataset,
                         model_cfg: mmcv.Config,
                         codebase: Codebase,
                         metrics: Optional[str] = None,
                         out: Optional[str] = None,
                         metric_options: Optional[dict] = None,
                         format_only: bool = False):
    """Perform post-processing to predictions of model.

    Args:
        outputs (list): A list of predictions of model inference.
        dataset (Dataset): Input dataset to run test.
        model_cfg (mmcv.Config): The model config.
        codebase (Codebase): Specifying codebase type.
        metrics (str): Evaluation metrics, which depends on
            the codebase and the dataset, e.g., "bbox", "segm", "proposal"
            for COCO, and "mAP", "recall" for PASCAL VOC in mmdet; "accuracy",
            "precision", "recall", "f1_score", "support" for single label
            dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for
            multi-label dataset in mmcls. Defaults is `None`.
        out (str): Output result file in pickle format, defaults to `None`.
        metric_options (dict): Custom options for evaluation, will be kwargs
            for dataset.evaluate() function. Defaults to `None`.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server. Defaults
            to `False`.
    """
    if codebase == Codebase.MMCLS:
        if metrics:
            results = dataset.evaluate(outputs, metrics, metric_options)
            for k, v in results.items():
                print(f'\n{k} : {v:.2f}')
        else:
            warnings.warn('Evaluation metrics are not specified.')
            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [dataset.CLASSES[lb] for lb in pred_label]
            results = {
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_class': pred_class
            }
            if not out:
                print('\nthe predicted result for the first element is '
                      f'pred_score = {pred_score[0]:.2f}, '
                      f'pred_label = {pred_label[0]} '
                      f'and pred_class = {pred_class[0]}. '
                      'Specify --out to save all results to files.')
        if out:
            print(f'\nwriting results to {out}')
            mmcv.dump(results, out)

    elif codebase == Codebase.MMDET:
        if out:
            print(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)
        kwargs = {} if metric_options is None else metric_options
        if format_only:
            dataset.format_results(outputs, **kwargs)
        if metrics:
            eval_kwargs = model_cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=metrics, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

    elif codebase == Codebase.MMSEG:
        if out:
            print(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)
        kwargs = {} if metric_options is None else metric_options
        if format_only:
            dataset.format_results(outputs, **kwargs)
        if metrics:
            dataset.evaluate(outputs, metrics, **kwargs)

    elif codebase == Codebase.MMOCR:
        if out:
            print(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)
        kwargs = {} if metric_options is None else metric_options
        if format_only:
            dataset.format_results(outputs, **kwargs)
        if metrics:
            eval_kwargs = model_cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=metrics, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

    elif codebase == Codebase.MMEDIT:
        if out:
            print(f'\nwriting results to {out}')
            mmcv.dump(outputs, out)
        # The Dataset doesn't need metrics
        print('\n')
        # print metrics
        stats = dataset.evaluate(outputs)
        for stat in stats:
            print('Eval-{}: {}'.format(stat, stats[stat]))

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase.value}')
