# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmdeploy.core.rewriters.rewriter_utils import RewriterRegistry
from mmdeploy.utils.constants import Backend


def test_check_backend():
    with pytest.raises(Exception):
        registry = RewriterRegistry()
        registry._check_backend(Backend.ONNXRUNTIME.value)


def test_add_backend():
    registry = RewriterRegistry()
    registry.add_backend(Backend.ONNXRUNTIME.value)
    assert Backend.ONNXRUNTIME.value in registry._rewrite_records
    assert Backend.DEFAULT.value in registry._rewrite_records
    assert Backend.TENSORRT.value not in registry._rewrite_records


def test_register_object():
    registry = RewriterRegistry()

    @registry.register_object('add', backend=Backend.DEFAULT.value)
    def add(a, b):
        return a + b

    records = registry._rewrite_records[Backend.DEFAULT.value]
    assert records is not None
    assert records['add'] is not None
    assert records['add']['_object'] is not None
    add_func = records['add']['_object']
    assert add_func(123, 456) == 123 + 456


def test_get_records():
    registry = RewriterRegistry()
    registry.add_backend(Backend.TENSORRT.value)

    @registry.register_object('add', backend=Backend.DEFAULT.value)
    def add(a, b):
        return a + b

    @registry.register_object('minus', backend=Backend.DEFAULT.value)
    def minus(a, b):
        return a - b

    @registry.register_object('add', backend=Backend.TENSORRT.value)
    def fake_add(a, b):
        return a * b

    default_records = dict(registry.get_records(Backend.DEFAULT.value))
    assert default_records['add']['_object'](1, 1) == 2
    assert default_records['minus']['_object'](1, 1) == 0

    tensorrt_records = dict(registry.get_records(Backend.TENSORRT.value))
    assert tensorrt_records['add']['_object'](1, 1) == 1
    assert tensorrt_records['minus']['_object'](1, 1) == 0
