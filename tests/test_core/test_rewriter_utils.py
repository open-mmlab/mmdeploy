# Copyright (c) OpenMMLab. All rights reserved.
import mmdeploy
import mmdeploy.core.rewriters.rewriter_utils as rewriter_utils
from mmdeploy.core.rewriters.rewriter_utils import (BackendChecker,
                                                    RewriterRegistry,
                                                    collect_env)
from mmdeploy.utils.constants import IR, Backend


def test_collect_env():
    env_dict = collect_env(Backend.ONNXRUNTIME, IR.ONNX, version='1.0')
    assert env_dict['backend'] == Backend.ONNXRUNTIME
    assert env_dict['ir'] == IR.ONNX
    assert env_dict['version'] == '1.0'
    assert env_dict['mmdeploy'] == mmdeploy.__version__


class TestChecker:
    env = collect_env(Backend.ONNXRUNTIME, IR.ONNX)

    def test_backend_checker(self):
        true_checker = rewriter_utils.BackendChecker(Backend.ONNXRUNTIME)
        assert true_checker.check(self.env) is True

        false_checker = rewriter_utils.BackendChecker(Backend.TENSORRT)
        assert false_checker.check(self.env) is False

    def test_ir_checker(self):
        true_checker = rewriter_utils.IRChecker(IR.ONNX)
        assert true_checker.check(self.env) is True

        false_checker = rewriter_utils.IRChecker(IR.TORCHSCRIPT)
        assert false_checker.check(self.env) is False

    def test_lib_version_checker(self):
        true_checker = rewriter_utils.LibVersionChecker(
            'mmdeploy', mmdeploy.__version__, mmdeploy.__version__)
        assert true_checker.check(self.env) is True

        false_checker = rewriter_utils.LibVersionChecker(
            'mmdeploy', max_version='0.0.0')
        assert false_checker.check(self.env) is False


def test_register_object():
    registry = RewriterRegistry()
    checker = rewriter_utils.BackendChecker(Backend.ONNXRUNTIME)

    @registry.register_object(
        'add',
        backend=Backend.DEFAULT.value,
        ir=IR.DEFAULT,
        extra_checkers=checker)
    def add(a, b):
        return a + b

    records = registry._rewrite_records
    assert records is not None
    assert records['add'] is not None
    assert isinstance(records['add'][0]['_checkers'], list)
    assert isinstance(records['add'][0]['_checkers'][0], BackendChecker)
    assert records['add'][0]['_object'] is not None
    add_func = records['add'][0]['_object']
    assert add_func(123, 456) == 123 + 456


def test_get_records():
    registry = RewriterRegistry()

    @registry.register_object(
        'get_num', backend=Backend.ONNXRUNTIME.value, ir=IR.ONNX)
    def get_num_1():
        return 1

    @registry.register_object(
        'get_num', backend=Backend.ONNXRUNTIME.value, ir=IR.TORCHSCRIPT)
    def get_num_2():
        return 2

    @registry.register_object(
        'get_num', backend=Backend.TENSORRT.value, ir=IR.ONNX)
    def get_num_3():
        return 3

    @registry.register_object(
        'get_num', backend=Backend.TENSORRT.value, ir=IR.TORCHSCRIPT)
    def get_num_4():
        return 4

    @registry.register_object(
        'get_num', backend=Backend.DEFAULT.value, ir=IR.DEFAULT)
    def get_num_5():
        return 5

    records = dict(
        registry.get_records(collect_env(Backend.ONNXRUNTIME, IR.ONNX)))
    assert records['get_num']['_object']() == 1

    records = dict(
        registry.get_records(collect_env(Backend.ONNXRUNTIME, IR.TORCHSCRIPT)))
    assert records['get_num']['_object']() == 2

    records = dict(
        registry.get_records(collect_env(Backend.TENSORRT, IR.ONNX)))
    assert records['get_num']['_object']() == 3

    records = dict(
        registry.get_records(collect_env(Backend.TENSORRT, IR.TORCHSCRIPT)))
    assert records['get_num']['_object']() == 4

    records = dict(registry.get_records(collect_env(Backend.NCNN, IR.ONNX)))
    assert records['get_num']['_object']() == 5
