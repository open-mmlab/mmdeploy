# 如何支持新的后端

MMDeploy 支持了许多后端推理引擎，但我们依然非常欢迎新后端的贡献。在本教程中，我们将介绍在 MMDeploy 中支持新后端的一般过程。

## 必要条件

在对 MMDeploy 添加新的后端引擎之前，需要先检查所要支持的新后端是否符合一些要求:

- 后端必须能够支持 ONNX 作为 IR。
- 如果后端需要“.onnx”文件以外的模型文件或权重文件，则需要添加将“.onnx”文件转换为模型文件或权重文件的转换工具，该工具可以是 Python API、脚本或可执行程序。
- 强烈建议新后端可提供 Python 接口来加载后端文件和推理以进行验证。

`mmdeploy/backend` 目录下有许多已经接入的后端。如果在实现时存在任何困难，可以参考其中的代码实现。

## 创建 BackendParam

1. `BaseBackendParam` 是一个 dataclass 类，我们将模型转换与推理需要的所有数据打包在其中，方便后续的操作

   我们非常推荐您给 BackendParam 的实现提供一个 google 风格的 docsting，我们会根据 docstring 的内容生成一个命令行参数解析器，帮助创建命令行工具

   ```python
   # mmdeploy/backend/newengine/backend_manager.py
   from ..base import BaseBackendParam
   class NewEngineParam(BaseBackendParam):
       """Your first backend parameters.

       Args:
           work_dir (str): The working directory.
           file_name (str): File name of the serialized model. Postfix will be
                added automatically.
       """

       work_dir:str = None
       # FileNameDescriptor will add postfix to file name if it does not has one
       file_name: FileNameDescriptor = FileNameDescriptor(
           default=None, postfix='.model')
   ```

2. 给创建的 param 对象提供一个方法，查询转换后的文件名。如果转换后或生成多个文件，则返回一个 string 列表

   ```python
   # mmdeploy/backend/newengine/backend_manager.py
   class NewEngineParam(BaseBackendParam):

       ...

       def get_model_files(self) -> Union[List, str]:
           """get the model files."""
           return osp.join(self.work_dir, self.file_name)
   ```

## 环境检查

每个接入的后端以 BackendManager 对象作为入口，它提供了包括转换、推理、环境检查等功能。

```python
# mmdeploy/backend/newengine/backend_manager.py

from mmdeploy.ir.onnx import ONNXParam
from ..base import BaseBackendManager
@BACKEND_MANAGERS.register('newengine', param=NewEngineParam, ir_param=ONNXParam)
class NewEngineManager(BaseBackendManager):
```

如果你希望将接入的后端贡献给 MMDeploy，那么可以在 `mmdeploy/utils/constants.py` 中添加如下代码并接受我们真挚的感谢

```Python
# mmdeploy/utils/constants.py

class Backend(AdvancedEnum):
    # Take TensorRT as an example
    NEWENGINE = 'newengine'
```

在我们真正开始进行功能的接入之前，首先应该对当前环境进行检查以确保后续运算可以正确进行。因此我们需要如下接口：

- `is_available` 返回 bool 值，表示该后端在当前环境下可用
- `get_version` 返回该后端的版本信息
- `check_env` 提供更详细的当前环境信息，用于帮助用户配置环境

```python
# mmdeploy/backend/newengine/backend_manager.py
class NewEngineManager(BaseBackendManager):

    ...

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        return my_backend_is_available()

    @classmethod
    def get_version(cls) -> str:
        return my_backend_version()

    @classmethod
    def check_env(cls, log_callback: Callable = lambda _: _) -> str:
        log_callback('Check env of your backend!')
        return super().check_env(log_callback=log_callback)

```

## 支持后端转换

多数后端都会有自己的序列化模型格式，为了支持从中间表示到该格式的转换，需要提供两个函数：

```python
# mmdeploy/backend/newengine/backend_manager.py
class NewEngineManager(BaseBackendManager):

    ...

    @classmethod
    def to_backend(cls, ir_model: str, *args, **kwargs):
        # convert your model here

    @classmethod
    def to_backend_from_param(cls, ir_model: str, param: NewEngineParam):
        # convert the model with the backend param you have just defined
```

`to_backend` 用来将 IR 模型转换成后端需要的格式，我们不对输入参数做太多限制，可以根据后端的需要自由配置。
`to_backend_from_param` 使用上面章节实现的 BackendParam 对象来实现转换。可以从 param 中提取数据然后调用 `to_backend` 以复用代码。

## 支持后端推理

如果希望可以使用 python 进行精度验证，那么就需要实现一个 Wrapper 对象和对应的构建函数。Wrapper 对象对后端推理细节进行了封装，用户可以像使用 PyTorch 模型那样使用后端接口。Wrapper 的输入输出为 Tensor 的 dict 对象。

```python
# mmdeploy/backend/newengine/wrapper.py
from mmdeploy.utils import Backend
from ..base import BACKEND_WRAPPER, BaseWrapper

@BACKEND_WRAPPER.register_module(Backend.NEWENGINE.value)
class NewEngineWrapper(BaseWrapper):

    def __init__(self, backend_file, other_arguments):
        # initialize the backend model here

    def forward(self, inputs) -> Dict[str, Tensor]:
        # perform inference here
```

实现了 Wrapper 以后，就可以在 backend manager 中添加构建函数：

```python
# mmdeploy/backend/newengine/backend_manager.py
class NewEngineManager(BaseBackendManager):

    ...

    @classmethod
    def build_wrapper(cls, backend_file, other_arguments):
        from .wrapper import NewEngineWrapper
        return NewEngineWrapper(backend_file, other_arguments)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        backend_file = get_backend_file_from_param(param)
        other_arguments = get_other_arguments_from_param(param)
        return cls.build_wrapper(backend_file, other_arguments)
```

## 命令行工具

如果希望将接入的后端作为一个命令行工具使用，可以实现 `parse_args` 接口：

```python
# mmdeploy/backend/newengine/backend_manager.py
class NewEngineManager(BaseBackendManager):

    ...

    @classmethod
    @contextlib.contextmanager
    def parse_args(cls,
                   parser: ArgumentParser,
                   args: Optional[List[str]] = None):

        # setup parser
        parser.add_argument(
            '--onnx-path', required=True, help='ONNX model path.')
        NewEngineParam.add_arguments(parser)

        parsed_args = parser.parse_args(args)

        yield parsed_args

        # convert model
        param = NewEngineParam(
            work_dir=parsed_args.work_dir,
            file_name=parsed_args.file_name)

        cls.to_backend_from_param(parsed_args.onnx_path, param)
```

`NewEngineParam.add_arguments(parser)` 会根据之前在 `NewEngineParam` 中添加的 docstring 信息自动生成解析器的 arguments。方便我们更快实现功能。

我们只要实现下面几行代码，就可以完成该工具：

```python
# console_tool.py
from ... import NewEngineManager

if __name__ == '__main__'
    NewEngineManager.main()
```

使用效果如下

```bash
python console_tool.py -h

# usage: test_dataclass.py convert [-h] --onnx-path ONNX_PATH
#                                  [--work-dir WORK_DIR] [--file-name FILE_NAME]
#
# build TensorRTParam
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --onnx-path ONNX_PATH
#                         ONNX model path.
#   --work-dir WORK_DIR   The working directory.
#   --file-name FILE_NAME
#                         File name of the serialized model. Postfix will
#                         beadded automatically.

```

## 单元测试

开发单元测试是一个好习惯，可以为功能维护以及更新带来便利。可以参考 `tests/test_backend` 添加自己的后端单元测试。

## deploy.py 支持

当前 MMDeploy 的总接口为 `tools/deploy.py`。它需要一个配置文件来实现转换、推理任务。如果希望接口的后端能够使用该接口，那么后端应该提供自己的配置文件。

配置文件的写法可以参考 [config tutorial](../02-how-to-run/write_config.md)， 这里不做赘述。假设我们为新添加的后端提供了如下配置文件：

```python
_base_ = ['./classification_dynamic.py']
codebase_config = dict(type='mmcls', task='Classification')
onnx_config = dict(input_shape=None)
backend_config = dict(type='newengine')
```

需要在 backend manager 对象中添加解析函数，通过 config 生成 BackendParam 对象

```python
# mmdeploy/backend/newengine/backend_manager.py
class NewEngineManager(BaseBackendManager):

    ...

    @classmethod
    def build_param_from_config(cls,
                                config: Any,
                                work_dir: str,
                                backend_files: Sequence[str] = None,
                                **kwargs) -> NewEngineParam:

        # create NewEngineParam with the config
```

如果能够正确配置上述步骤，那么恭喜你，你已经完成了后端接入，可以开始在 MMDeploy 中享受新的推理后端！
