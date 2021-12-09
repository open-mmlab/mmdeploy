# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import Dataset


def is_can_sort_dataset(dataset: Dataset) -> bool:
    """Checking for the possibility of sorting the dataset by fields 'height'
    and 'width'.

    Args:
        dataset (Dataset): The dataset.

    Returns:
        bool: Is it possible or not to sort the dataset.
    """
    is_sort_possible = \
        hasattr(dataset, 'data_infos') and \
        dataset.data_infos and \
        all(key in dataset.data_infos[0] for key in ('height', 'width'))
    return is_sort_possible


def sort_dataset(dataset: Dataset) -> Dataset:
    """Sorts the dataset by image height and width.

    Args:
        dataset (Dataset): The dataset.

    Returns:
        Dataset: Sorted dataset.
    """
    sort_data_infos = sorted(
        dataset.data_infos, key=lambda e: (e['height'], e['width']))
    sort_img_ids = [e['id'] for e in sort_data_infos]
    dataset.data_infos = sort_data_infos
    dataset.img_ids = sort_img_ids
    return dataset
