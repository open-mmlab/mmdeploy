# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.utils.dataset import is_can_sort_dataset, sort_dataset


class DummyDataset():

    def __init__(self, data_infos=None):
        if data_infos:
            self.data_infos = data_infos


emtpy_dataset = DummyDataset()
dataset = DummyDataset([{
    'id': 0,
    'height': 0,
    'width': 0
}, {
    'id': 1,
    'height': 1,
    'width': 1
}, {
    'id': 2,
    'height': 1,
    'width': 0
}, {
    'id': 3,
    'height': 0,
    'width': 1
}])


class TestIsCanSortDataset:

    def test_is_can_sort_dataset_false(self):
        assert not is_can_sort_dataset(emtpy_dataset)

    def test_is_can_sort_dataset_True(self):
        assert is_can_sort_dataset(dataset)


def test_sort_dataset():
    result_dataset = sort_dataset(dataset)
    assert result_dataset.data_infos == [{
        'id': 0,
        'height': 0,
        'width': 0
    }, {
        'id': 3,
        'height': 0,
        'width': 1
    }, {
        'id': 2,
        'height': 1,
        'width': 0
    }, {
        'id': 1,
        'height': 1,
        'width': 1
    }]
    assert result_dataset.img_ids == [0, 3, 2, 1]
