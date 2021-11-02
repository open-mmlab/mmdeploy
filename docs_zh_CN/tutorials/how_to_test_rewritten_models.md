# 如何测试重写模型

当你使用我们的[重写器](how_to_support_new_models.md)创建了一个重写模型后，最好为该模型写一个单元测试以确保模型重写能够生效。一般情况下，我们需要获取原模型和重写模型的输出，并对输出做比对。原模型的输出可以通过直接调用前向推理函数获得，而获取重写模型输出的方式取决于重写的复杂程度。

## 测试小幅改动的重写模型

如果对重写模型的修改比较小（例如只是修改了一两个变量的行为，而不对其他地方产生影响），你可以先构造输入参数，之后直接在 `RewriteContext` 中执行模型推理并检查输出结果。

```python
# mmcls.models.classfiers.base.py
class BaseClassifier(BaseModule, metaclass=ABCMeta):
    def forward(self, img, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

# Custom rewritten function
@FUNCTION_REWRITER.register_rewriter(
    'mmcls.models.classifiers.BaseClassifier.forward', backend='default')
def forward_of_base_classifier(ctx, self, img, *args, **kwargs):
    """Rewrite `forward` for default backend."""
    return self.simple_test(img, {})
```

在这个例子中，我们只改变了 `foward` 函数调用的函数。我们可以用如下的测试函数来测试这个重写模型：

```python
def test_baseclassfier_forward():
    input = torch.rand(1)
    from mmcls.models.classifiers import BaseClassifier
    class DummyClassifier(BaseClassifier):

        def __init__(self, init_cfg=None):
            super().__init__(init_cfg=init_cfg)

        def extract_feat(self, imgs):
            pass

        def forward_train(self, imgs):
            return 'train'

        def simple_test(self, img, tmp, **kwargs):
            return 'simple_test'

    model = DummyClassifier().eval()

    model_output = model(input)
    with RewriterContext(cfg=dict()), torch.no_grad():
        backend_output = model(input)

    assert model_output == 'train'
    assert backend_output == 'simple_test'
```

在该测试函数中，我们通过构造一个 `BaseClassifier` 的派生类来测试重写模型是否能够在重写的上下文中生效。我们通过直接调用 `model(input)` 来获取原模型的输出并通过在 `RewriteContext` 中调用 `model(input)` 来获取重写模型的输出。最后，我们可以通过断言输出的结果的值来验证结果的正确性。

## 测试大幅改动的重写模型

在第一个例子中，重写的输出是在 Python 中生成的。有些时候，我们会对原模型做一些较大程度的改动（比如为生成正确的计算图而去掉分支语句）。即使重写模型运行在 Python 上得到的结果是正确的，我们不能保证重写模型在后端上也能够如期运行。因此，我们需要在后端中测试重写模型。

```python
# Custom rewritten function
@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.segmentors.BaseSegmentor.forward')
def forward_of_base_segmentor(ctx, self, img, img_metas=None, **kwargs):
    if img_metas is None:
        img_metas = {}
    assert isinstance(img_metas, dict)
    assert isinstance(img, torch.Tensor)

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    img_shape = img.shape[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    img_metas['img_shape'] = img_shape
    return self.simple_test(img, img_metas, **kwargs)

```

该重写函数的行为比较复杂，我们应该用如下方法进行测试：

```python
def test_basesegmentor_forward():
    from mmdeploy.utils.test import (WrapModel, get_model_outputs,
                                    get_rewrite_outputs)

    segmentor = get_model()
    segmentor.cpu().eval()

    # 准备数据
    # ...

    # 获取原模型的输出
    model_inputs = {
        'img': [imgs],
        'img_metas': [img_metas],
        'return_loss': False
    }
    model_outputs = get_model_outputs(segmentor, 'forward', model_inputs)

    # 获取重写模型的输出
    wrapped_model = WrapModel(segmentor, 'forward', img_metas = None, return_loss = False)
    rewrite_inputs = {'img': imgs}
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        # 如果后端插件成功运行的话，重写输出是后端生成的
        rewrite_outputs = torch.tensor(rewrite_outputs)
        model_outputs = torch.tensor(model_outputs)
        model_outputs = model_outputs.unsqueeze(0).unsqueeze(0)
        assert torch.allclose(rewrite_outputs, model_outputs)
    else:
        # 否则，重写输出是 Python 生成的
        assert rewrite_outputs is not None
```

我们提供了一些实用函数来测试重写函数。首先，你可以构造一个模型并调用 `get_model_outputs` 获得原模型的输出。之后，你可以用 `WrapModel` 来封装重写函数。这里的 `WrapModel` 可以看成是一个偏函数（能够预设好重写函数的控制参数，使得重写函数的唯一参数是输入张量）。把 `WrapModel` 的实例传入 `get_rewrite_outputs` 就能获取重写模型的输出。`get_rewrite_outputs` 有两个返回值，一个表示重写输出的内容，另一个表示输出是否来自后端。由于我们不能假设每个用户都安装了后端库，我们需要检查重写结果是在 Python 还是在后端里生成的。单元测试必须覆盖这两种情况。最后，我们需要比较原模型和重写模型的输出，这一比较可以通过调用来 `torch.allclose` 来轻松完成。

## 注意事项

TODO： 添加api文档的链接

若想了解测试实用函数的完整用法，请参考我们的 [api 文档](单元测试api链接)。
