# How to add test units for backend ops

This tutorial introduces how to add unit test for backend ops. When you add a custom op under `backend_ops`, you need to add the corresponding test unit. Test units of ops are included in `tests/test_ops/test_ops.py`.

## Prerequisite

- `Compile new ops`: After adding a new custom op, needs to recompile the relevant backend, referring to [build.md](../01-how-to-build/build_from_source.md).

## 1. Add the test program test_XXXX()

You can put unit test for ops in `tests/test_ops/`. Usually, the following program template can be used for your custom op.

### example of ops unit test

```python
@pytest.mark.parametrize('backend', [TEST_TENSORRT, TEST_ONNXRT])        # 1.1 backend test class
@pytest.mark.parametrize('pool_h,pool_w,spatial_scale,sampling_ratio',   # 1.2 set parameters of op
                         [(2, 2, 1.0, 2), (4, 4, 2.0, 4)])               # [（# Examples of op test parameters）,...]
def test_roi_align(backend,
                   pool_h,                                               # set parameters of op
                   pool_w,
                   spatial_scale,
                   sampling_ratio,
                   input_list=None,
                   save_dir=None):
    backend.check_env()

    if input_list is None:
        input = torch.rand(1, 1, 16, 16, dtype=torch.float32)            # 1.3 op input data initialization
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
    else:
        input = torch.tensor(input_list[0], dtype=torch.float32)
        single_roi = torch.tensor(input_list[1], dtype=torch.float32)

    from mmcv.ops import roi_align

    def wrapped_function(torch_input, torch_rois):                       # 1.4 initialize op model to be tested
        return roi_align(torch_input, torch_rois, (pool_w, pool_h),
                         spatial_scale, sampling_ratio, 'avg', True)

    wrapped_model = WrapFunction(wrapped_function).eval()

    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11): # 1.5 call the backend test class interface
        backend.run_and_validate(
            wrapped_model, [input, single_roi],
            'roi_align',
            input_names=['input', 'rois'],
            output_names=['roi_feat'],
            save_dir=save_dir)
```

### 1.1 backend test class

We provide some functions and classes for difference backends, such as `TestOnnxRTExporter`, `TestTensorRTExporter`, `TestNCNNExporter`.

### 1.2 set parameters of op

Set some parameters of op, such as ’pool_h‘, ’pool_w‘, ’spatial_scale‘, ’sampling_ratio‘ in roi_align. You can set multiple parameters to test op.

### 1.3 op input data initialization

Initialization required input data.

### 1.4 initialize op model to be tested

The model containing custom op usually has two forms.

- `torch model`: Torch model with custom operators. Python code related to op is required, refer to `roi_align` unit test.
- `onnx model`: Onnx model with custom operators. Need to call onnx api to build, refer to `multi_level_roi_align` unit test.

### 1.5 call the backend test class interface

Call the backend test class `run_and_validate` to run and verify the result output by the op on the backend.

```python
    def run_and_validate(self,
                         model,
                         input_list,
                         model_name='tmp',
                         tolerate_small_mismatch=False,
                         do_constant_folding=True,
                         dynamic_axes=None,
                         output_names=None,
                         input_names=None,
                         expected_result=None,
                         save_dir=None):
```

#### Parameter Description

- `model`: Input model to be tested and it can be torch model or any other backend model.
- `input_list`: List of test data, which is mapped to the order of input_names.
- `model_name`: The name of the model.
- `tolerate_small_mismatch`: Whether to allow small errors in the verification of results.
- `do_constant_folding`: Whether to use constant light folding to optimize the model.
- `dynamic_axes`: If you need to use dynamic dimensions, enter the dimension information.
- `output_names`: The node name of the output node.
- `input_names`: The node name of the input node.
- `expected_result`: Expected ground truth values for verification.
- `save_dir`: The folder used to save the output files.

## 2. Test Methods

Use pytest to call the test function to test ops.

```bash
pytest tests/test_ops/test_ops.py::test_XXXX
```
