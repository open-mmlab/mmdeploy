# How to support new backends

MMDeploy supports a number of backend engines. We welcome the contribution of new backends. In this tutorial, we will introduce the general procedures to support a new backend in MMDeploy.

## Prerequisites

Before contributing the codes, there are some requirements for the new backend that need to be checked:

- The backend must support ONNX or Torchscript as IR.
- If the backend requires model files or weight files other than a ".onnx" file, a conversion tool that converts the ".onnx" file to model files and weight files is required. The tool can be a Python API, a script, or an executable program.
- It is highly recommended that the backend provides a Python interface to load the backend files and inference for validation.

There are a lot of backends in `mmdeploy/backend`. Feel free to read the codes if you meet any problems.

## Create BackendParam

1. `BaseBackendParam` is a dataclass that we used to package everything parameters we need to do the conversion or inference.

   It is highly recommend to add google style docstring for your param. We will generate arguments for the `ArgumentParser` to ease the console tools.

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

2. Add a method to tell us the backend model names.

   ```python
   # mmdeploy/backend/newengine/backend_manager.py
   class NewEngineParam(BaseBackendParam):

       ...

       def get_model_files(self) -> Union[List, str]:
           """get the model files."""
           return osp.join(self.work_dir, self.file_name)
   ```

## Check environment

A backend manager is the entry to everything about the backend. Assume your backend convert model from onnx. You can create the manager below:

```python
# mmdeploy/backend/newengine/backend_manager.py

from mmdeploy.ir.onnx import ONNXParam
from ..base import BaseBackendManager
@BACKEND_MANAGERS.register('newengine', param=NewEngineParam, ir_param=ONNXParam)
class NewEngineManager(BaseBackendManager):
```

If you decide to contribute the backend manager to MMDeploy, Do not forget to add enumrate in `mmdeploy/utils/constants.py`

```Python
# mmdeploy/utils/constants.py

class Backend(AdvancedEnum):
    # Take TensorRT as an example
    NEWENGINE = 'newengine'
```

Before we do anything with the backend. We want to make sure everything is fine. Let's add some method to check the environment.

- `is_available` return bool indicate that the backend manager is available on current device.
- `get_version` return the backend version information.
- `check_env` provide detail information about the backend.

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

## Support backend conversion

Most backend has it's own serialize format. To support the conversion, Two method is required in backend manager:

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

`to_backend` convert the model to the your backend. There is no limitation on the arguments of the method. It is up to you.

`to_backend_from_param` accept a serialized IR file and the backend param you have just defined. You can extract fields from the backend param and convert model with `to_backend`.

## Support backend inference

It would be cool if we can evaluate the backend model with python. Create a backend wrapper so we can perform inference with the backend and hide the detail. The inputs/outputs of the wrapper is a `dict` of `torch.Tensor`.

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

Once you have a backend wrapper, add method in manager to build it.

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

## Console argument parser

What if you want to use the backend manager as a console tool. You can implement `parse_args` in the backend manager:

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

`NewEngineParam.add_arguments(parser)` would add the arguments to the parser according to the docstring. You can create your console tool as:

```python
# console_tool.py
from ... import NewEngineManager

if __name__ == '__main__'
    NewEngineManager.main()
```

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

## Unit Test

Develop the unit test can help us test and maintain the backend support. Read scripts in `tests/test_backend` for model details.

## Deploy config support

MMDeploy provide config files to describe the task you want to perform. Our main entry `tools/deploy.py` require the config file to perform the conversion and inference. If you hope the new backend can be use by `tools/deploy.py`, you would need a config too.

The config is a dictionary that composed of `model_config`, `ir_config`, `backend_config`... And they can be inherited by `__base__ = [...]`.
Create a config for the codebase you want to use the new backend. mmclassification as example:

```python
_base_ = ['./classification_dynamic.py']
codebase_config = dict(type='mmcls', task='Classification')
onnx_config = dict(input_shape=None)
backend_config = dict(type='newengine')
```

Read [config tutorial](../02-how-to-run/write_config.md) for more detail about the config file.

Then add a convert tool in the backend manager:

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

Now we have finish all steps. Enjoy the new backend and MMDeploy!
