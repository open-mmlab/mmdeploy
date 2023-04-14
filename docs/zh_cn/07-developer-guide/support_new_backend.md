# 如何支持新的后端

MMDeploy 支持了许多后端推理引擎，但我们依然非常欢迎新后端的贡献。在本教程中，我们将介绍在 MMDeploy 中支持新后端的一般过程。

## 必要条件

在对 MMDeploy 添加新的后端引擎之前，需要先检查所要支持的新后端是否符合一些要求:

- 后端必须能够支持 ONNX 作为 IR。
- 如果后端需要“.onnx”文件以外的模型文件或权重文件，则需要添加将“.onnx”文件转换为模型文件或权重文件的转换工具，该工具可以是 Python API、脚本或可执行程序。
- 强烈建议新后端可提供 Python 接口来加载后端文件和推理以进行验证。

## 支持后端转换

MMDeploy 中的后端必须支持 ONNX，因此后端能直接加载“.onnx”文件，或者使用转换工具将“.onnx”转换成自己的格式。在本节中，我们将介绍支持后端转换的步骤。

1. 在 `mmdeploy/utils/constants.py` 文件中添加新推理后端变量，以表示支持的后端名称。

   **示例**：

   ```Python
   # mmdeploy/utils/constants.py

   class Backend(AdvancedEnum):
       # 以现有的TensorRT为例
       TENSORRT = 'tensorrt'
   ```

2. 在 `mmdeploy/backend/` 目录下添加相应的库(一个包括 `__init__.py` 的文件夹),例如， `mmdeploy/backend/tensorrt` 。在 `__init__.py` 中，必须有一个名为 `is_available` 的函数检查用户是否安装了后端库。如果检查通过，则将加载库的剩余文件。

   **例子**:

   ```Python
   # mmdeploy/backend/tensorrt/__init__.py

   def is_available():
       return importlib.util.find_spec('tensorrt') is not None


   if is_available():
       from .utils import from_onnx, load, save
       from .wrapper import TRTWrapper

       __all__ = [
           'from_onnx', 'save', 'load', 'TRTWrapper'
       ]
   ```

3. 在 `configs/_base_/backends` 目录中创建一个配置文件(例如， `configs/_base_/backends/tensorrt.py` )。如果新后端引擎只是将“.onnx”文件作为输入，那么新的配置可以很简单,对应配置只需包含一个表示后端名称的字段(但也应该与 `mmdeploy/utils/constants.py` 中的名称相同)。

   **例子**

   ```python
   backend_config = dict(type='tensorrt')
   ```

   但如果后端需要其他文件，则从“.onnx”文件转换为后端文件所需的参数也应包含在配置文件中。

   **例子**

   ```Python

   backend_config = dict(
       type='tensorrt',
       common_config=dict(
           fp16_mode=False, max_workspace_size=0))
   ```

   在拥有一个基本的后端配置文件后，您已经可以通过继承轻松构建一个完整的部署配置。有关详细信息，请参阅我们的[配置教程](../02-how-to-run/write_config.md)。下面是一个例子：

   ```Python
   _base_ = ['../_base_/backends/tensorrt.py']

   codebase_config = dict(type='mmpretrain', task='Classification')
   onnx_config = dict(input_shape=None)
   ```

4. 如果新后端需要模型文件或权重文件而不是“.onnx”文件，则需要在相应的文件夹中创建一个 `onnx2backend.py` 文件(例如,创建 `mmdeploy/backend/tensorrt/onnx2tensorrt.py` )。然后在文件中添加一个转换函数`onnx2backend`。该函数应将给定的“.onnx”文件转换为给定工作目录中所需的后端文件。对函数的其他参数和实现细节没有要求，您可以使用任何工具进行转换。下面有些例子：

   **使用python脚本**

   ```Python
   def onnx2openvino(input_info: Dict[str, Union[List[int], torch.Size]],
                     output_names: List[str], onnx_path: str, work_dir: str):

       input_names = ','.join(input_info.keys())
       input_shapes = ','.join(str(list(elem)) for elem in input_info.values())
       output = ','.join(output_names)

       mo_args = f'--input_model="{onnx_path}" '\
                 f'--output_dir="{work_dir}" ' \
                 f'--output="{output}" ' \
                 f'--input="{input_names}" ' \
                 f'--input_shape="{input_shapes}" ' \
                 f'--disable_fusing '
       command = f'mo.py {mo_args}'
       mo_output = run(command, stdout=PIPE, stderr=PIPE, shell=True, check=True)
   ```

   **使用可执行文件**

   ```Python
   def onnx2ncnn(onnx_path: str, work_dir: str):
       onnx2ncnn_path = get_onnx2ncnn_path()
       save_param, save_bin = get_output_model_file(onnx_path, work_dir)
       call([onnx2ncnn_path, onnx_path, save_param, save_bin])\
   ```

5. 在 `mmdeploy/apis` 中创建新后端库并声明对应 APIs

   **例子**

   ```Python
   # mmdeploy/apis/ncnn/__init__.py

   from mmdeploy.backend.ncnn import is_available

   __all__ = ['is_available']

   if is_available():
       from mmdeploy.backend.ncnn.onnx2ncnn import (onnx2ncnn,
                                                    get_output_model_file)
       __all__ += ['onnx2ncnn', 'get_output_model_file']
   ```

   从 BaseBackendManager 派生类，实现 `to_backend` 类方法。

   **例子**

   ```Python
   @classmethod
    def to_backend(cls,
                   ir_files: Sequence[str],
                   deploy_cfg: Any,
                   work_dir: str,
                   log_level: int = logging.INFO,
                   device: str = 'cpu',
                   **kwargs) -> Sequence[str]:
        return ir_files
   ```

6. 将 OpenMMLab 的模型转换后(如有必要)并在后端引擎上进行推理。如果在测试时发现一些不兼容的算子，可以尝试按照[重写器教程](support_new_model.md)为后端重写原始模型或添加自定义算子。

7. 为新后端引擎代码添加相关注释和单元测试:).

## 支持后端推理

尽管后端引擎通常用C/C++实现，但如果后端提供Python推理接口，则测试和调试非常方便。我们鼓励贡献者在MMDeploy的Python接口中支持新后端推理。在本节中，我们将介绍支持后端推理的步骤。

1. 添加一个名为 `wrapper.py` 的文件到 `mmdeploy/backend/{backend}` 中相应后端文件夹。例如， `mmdeploy/backend/tensorrt/wrapper` 。此模块应实现并注册一个封装类，该类继承 `mmdeploy/backend/base/base_wrapper.py` 中的基类 `BaseWrapper` 。

   **例子**

   ```Python
   from mmdeploy.utils import Backend
   from ..base import BACKEND_WRAPPER, BaseWrapper

   @BACKEND_WRAPPER.register_module(Backend.TENSORRT.value)
   class TRTWrapper(BaseWrapper):
   ```

2. 封装类可以在函数 `__init__` 中初始化引擎以及在 `forward` 函数中进行推理。请注意，该 `__init__` 函数必须接受一个参数 `output_names` 并将其传递给基类以确定输出张量的顺序。其中 `forward` 输入和输出变量应表示tensors的名称和值的字典。

3. 为了方便性能测试，该类应该定义一个 `execute` 函数，只调用后端引擎的推理接口。该 `forward` 函数应在预处理数据后调用 `execute` 函数。

   **例子**

   ```Python
   from mmdeploy.utils import Backend
   from mmdeploy.utils.timer import TimeCounter
   from ..base import BACKEND_WRAPPER, BaseWrapper

   @BACKEND_WRAPPER.register_module(Backend.ONNXRUNTIME.value)
   class ORTWrapper(BaseWrapper):

       def __init__(self,
                    onnx_file: str,
                    device: str,
                    output_names: Optional[Sequence[str]] = None):
           # Initialization
           #
           # ...
           super().__init__(output_names)

       def forward(self, inputs: Dict[str,
                                      torch.Tensor]) -> Dict[str, torch.Tensor]:
           # Fetch data
           # ...

           self.__ort_execute(self.io_binding)

   		# Postprocess data
           # ...

       @TimeCounter.count_time('onnxruntime')
       def __ort_execute(self, io_binding: ort.IOBinding):
   		# Only do the inference
           self.sess.run_with_iobinding(io_binding)
   ```

4. 从 `BaseBackendManager` 派生接口类，实现 `build_wrapper` 静态方法

   **例子**

   ```Python
        @BACKEND_MANAGERS.register('onnxruntime')
        class ONNXRuntimeManager(BaseBackendManager):
            @classmethod
            def build_wrapper(cls,
                              backend_files: Sequence[str],
                              device: str = 'cpu',
                              input_names: Optional[Sequence[str]] = None,
                              output_names: Optional[Sequence[str]] = None,
                              deploy_cfg: Optional[Any] = None,
                              **kwargs):
                from .wrapper import ORTWrapper
                return ORTWrapper(
                    onnx_file=backend_files[0],
                    device=device,
                    output_names=output_names)
   ```

5. 为新后端引擎代码添加相关注释和单元测试 :).

## 将MMDeploy作为第三方库时添加新后端

前面的部分展示了如何在 MMDeploy 中添加新的后端，这需要更改其源代码。但是，如果我们将 MMDeploy 视为第三方，则上述方法不再有效。为此，添加一个新的后端需要我们预先安装另一个名为 `aenum` 的包。我们可以直接通过`pip install aenum`进行安装。

成功安装 `aenum` 后，我们可以通过以下方式使用它来添加新的后端：

```python
from mmdeploy.utils.constants import Backend
from aenum import extend_enum

try:
    Backend.get('backend_name')
except Exception:
    extend_enum(Backend, 'BACKEND', 'backend_name')
```

我们可以在使用 MMDeploy 的重写逻辑之前运行上面的代码，这就完成了新后端的添加。
