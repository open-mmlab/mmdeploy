# Copyright (c) OpenMMLab. All rights reserved.
from tempfile import NamedTemporaryFile

import pytest


@pytest.fixture(scope='module')
def torch_model_2i2o():
    torch = pytest.importorskip('torch')

    class DummyModel(torch.nn.Module):

        def forward(self, x, y):
            return x + y, x - y

    yield DummyModel()


@pytest.fixture(scope='module')
def input_shape():
    yield [1, 3, 8, 8]


@pytest.fixture(scope='module')
def dummy_x(input_shape):
    torch = pytest.importorskip('torch')
    yield torch.rand(input_shape)


@pytest.fixture(scope='module')
def dummy_y(input_shape):
    torch = pytest.importorskip('torch')
    yield torch.rand(input_shape)


@pytest.fixture(scope='module')
def input_2i(dummy_x, dummy_y):
    yield dummy_x, dummy_y


@pytest.fixture(scope='module')
def output_2i2o(torch_model_2i2o, dummy_x, dummy_y):
    yield torch_model_2i2o(dummy_x, dummy_y)


@pytest.fixture(scope='module')
def input_names_2i():
    yield ['x', 'y']


@pytest.fixture(scope='module')
def output_names_2i2o():
    yield ['ox', 'oy']


@pytest.fixture(scope='module')
def input_dict_2i(input_2i, input_names_2i):
    yield dict(zip(input_names_2i, input_2i))


@pytest.fixture(scope='module')
def output_dict_2i2o(output_2i2o, output_names_2i2o):
    yield dict(zip(output_names_2i2o, output_2i2o))


@pytest.fixture(scope='module')
def input_shape_dict_2i(input_dict_2i):
    yield dict((name, val.shape) for name, val in input_dict_2i.items())


@pytest.fixture(scope='module')
def output_shape_dict_2i2o(output_dict_2i2o):
    yield dict((name, val.shape) for name, val in output_dict_2i2o.items())


@pytest.fixture(scope='module')
def dynamic_axes_2i():
    yield dict(x={0: 'b', 2: 'h', 3: 'w'}, y={0: 'b', 2: 'h', 3: 'w'})


@pytest.fixture(scope='module')
def onnx_model_static_2i2o(tmp_path, torch_model_2i2o, dummy_x, dummy_y,
                           input_names_2i, output_names_2i2o):
    torch = pytest.importorskip('torch')
    tmp_path = NamedTemporaryFile(suffix='.onnx').name
    torch.onnx.export(
        torch_model_2i2o, (dummy_x, dummy_y),
        tmp_path,
        input_names=input_names_2i,
        output_names=output_names_2i2o)

    yield tmp_path


@pytest.fixture(scope='module')
def onnx_model_dynamic_2i2o(torch_model_2i2o, dummy_x, dummy_y, input_names_2i,
                            output_names_2i2o, dynamic_axes_2i):
    torch = pytest.importorskip('torch')
    tmp_path = NamedTemporaryFile(suffix='.onnx').name
    torch.onnx.export(
        torch_model_2i2o, (dummy_x, dummy_y),
        tmp_path,
        input_names=input_names_2i,
        output_names=output_names_2i2o,
        dynamic_axes=dynamic_axes_2i)

    yield tmp_path


@pytest.fixture
def assert_forward():
    try:
        from torch.testing import assert_close as torch_assert_close
    except Exception:
        from torch.testing import assert_allclose as torch_assert_close

    def _impl(model, inputs, gts):
        outputs = model(inputs)
        for name in outputs:
            out = outputs[name]
            gt = gts[name]
            torch_assert_close(out, gt)

    return _impl
