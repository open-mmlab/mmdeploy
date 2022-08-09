# 为推理 ops 添加测试单元

本教程介绍如何为后端 ops 添加单元测试。在 backend_ops 目录下添加自定义 op 时，需要添加相应的测试单元。op 的单元测试在 `test/test_ops/test_ops.py` 中。

添加新的自定义 op 后，需要重新编译，引用 [build.md](../01-how-to-build/build_from_source.md) 。

## ops 单元测试样例

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

mmdeploy 支持的模型有两种格式：

- torch 模型：参考 roi_align 单元测试，必须要求 op 相关 Python 代码
- onnx 模型：参考 multi_level_roi_align 单元测试，需要调用 onnx api 进行构建

调用 `run_and_validate` 即可运行

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

|          参数           |                 说明                  |
| :---------------------: | :-----------------------------------: |
|          model          |           要测试的输入模型            |
|       input_list        | 测试数据列表，映射到input_names的顺序 |
| tolerate_small_mismatch |     是否允许验证结果出现精度误差      |
|   do_constant_folding   |           是否使用常量折叠            |
|      output_names       |             输出节点名字              |
|       input_names       |             输入节点名字              |
|     expected_result     |          期望的 ground truth          |
|        save_dir         |             结果保存目录              |

## 测试模型

用 `pytest` 调用 ops 测试

```bash
pytest tests/test_ops/test_ops.py::test_XXXX
```
