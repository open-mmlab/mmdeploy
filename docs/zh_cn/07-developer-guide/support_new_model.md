# 如何支持新的模型

我们提供了多种工具来支持模型转换

## 函数的重写器

PyTorch 神经网络是用 python 编写的，可以简化算法的开发。但与此同时 Python 的流程控制和第三方库会使得网络导出为中间语言的过程变得困难。为此我们提供了一个“MonKey path”工具将不支持的功能重写为另一个可支持中间语言导出的功能。下述是一个具体的使用例子：

```python
from mmdeploy.core import FUNCTION_REWRITER
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.repeat', backend='tensorrt')
def repeat_static(input, *size):
    ctx = FUNCTION_REWRITER.get_context()
    origin_func = ctx.origin_func
    if input.dim() == 1 and len(size) == 1:
        return origin_func(input.unsqueeze(0), *([1] + list(size))).squeeze(0)
    else:
        return origin_func(input, *size)
```

使用函数重写器是十分容易的，只需添加一个带参数的装饰器即可：

- `func_name`是需要被重载的函数，它可以是其他PyTorch 的函数或者是自定义的函数。模块中的方法也可以通过工具进行重载。
- `backend`是推理引擎。当模型被导入到引擎的时候，函数会被重载。如果没有给出，重载默认的参数就是重载的参数。如果后端的重载的参数不存在，将会按照预设的默认模式进行重载。
  当参数与原始的参数相同时，除了把上下文信息`ctx` 作为第一的参数外，上下文也提供了一些有用的信息，例如:部署的配置`ctx.cfg` 和原始的函数（已经被重载）`ctx.origin_func`。

可参照[这些样例代码](https://github.com/open-mmlab/mmdeploy/blob/main/mmdeploy/codebase/mmpretrain/models/backbones/shufflenet_v2.py)。

## 模型重载器

如果您想用另一个模块替换整个模块，我们还有另一个重载器，如下所示：

```python
@MODULE_REWRITER.register_rewrite_module(
    'mmagic.models.backbones.sr_backbones.SRCNN', backend='tensorrt')
class SRCNNWrapper(nn.Module):
    def __init__(self,
                 module,
                 cfg,
                 channels=(3, 64, 32, 3),
                 kernel_sizes=(9, 1, 5),
                 upscale_factor=4):
        super(SRCNNWrapper, self).__init__()
        self._module = module
        module.img_upsampler = nn.Upsample(
            scale_factor=module.upscale_factor,
            mode='bilinear',
            align_corners=False)
    def forward(self, *args, **kwargs):
        """Run forward."""
        return self._module(*args, **kwargs)
    def init_weights(self, *args, **kwargs):
        """Initialize weights."""
        return self._module.init_weights(*args, **kwargs)
```

就像函数重载器一样，可添加一个带参数的装饰器：

- `module_type` 要重载的模块类。
- `backend` 是推理引擎。当模型被导入到引擎的时候，函数会被重载。如果没有给出，重载默认的参数就是重载的参数。如果后端的重载的参数不存在，将会按照预设的默认模式进行重载。

网络中模块的所有实例都将替换为这个新类的实例。原始模块和部署配置将作为前两个参数进行传递。

## 符号函数重写

PyTorch 和 ONNX 之间的映射是通过 PyTorch 中的符号函数进行定义的。自定义符号函数可以帮助我们绕过一些推理引擎不支持的 ONNX 节点。

```python
@SYMBOLIC_REWRITER.register_symbolic('squeeze', is_pytorch=True)
def squeeze_default(g, self, dim=None):
    if dim is None:
        dims = []
        for i, size in enumerate(self.type().sizes()):
            if size == 1:
                dims.append(i)
    else:
        dims = [sym_help._get_const(dim, 'i', 'dim')]
    return g.op('Squeeze', self, axes_i=dims)
```

装饰器的参数

- `func_name`要添加符号的函数名称。如果是自定义的，请使用完整路径`torch.autograd.Function`。或者如果它是 PyTorch 内置函数，则只用写一个名称即可。
- `backend`是推理引擎。当模型被导入到引擎的时候，函数会被重载。如果没有给出，重载默认的参数就是重载的参数。如果后端的重载的参数不存在，将会按照预设的默认模式进行重载。
- 如果函数是 PyTorch 内置函数，则为True。
- `arg_descriptors` 符号函数参数的描述符，将被传递给`torch.onnx.symbolic_helper._parse_arg`。

就像函数重载器的`ctx`一样，第一个参数会提供上下文信息。上下文中了一些有用的信息，例如部署配置ctx.cfg和原始功能（已被重载）`ctx.origin_func`。请注意， `ctx.origin_func`只能在`is_pytorch==False`时使用。

[这里](https://github.com/open-mmlab/mmdeploy/tree/6420e2044515ff2052960c0f8bb9e351e6a7f2c2/mmdeploy/pytorch/symbolics)有很多实现可参考。
