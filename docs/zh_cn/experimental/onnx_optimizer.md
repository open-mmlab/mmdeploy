# ONNX export Optimizer

This is a tool to optimize ONNX model when exporting from PyTorch.

## Installation

Build MMDeploy with `torchscript` support:

```shell
export Torch_DIR=$(python -c "import torch;print(torch.utils.cmake_prefix_path + '/Torch')")

cmake \
    -DTorch_DIR=${Torch_DIR} \
    -DMMDEPLOY_TARGET_BACKENDS="${your_backend};torchscript" \
    .. # You can also add other build flags if you need

cmake --build . -- -j$(nproc) && cmake --install .
```

## Usage

```python
# import model_to_graph_custom_optimizer so we can hijack onnx.export
from mmdeploy.apis.onnx.optimizer import model_to_graph__custom_optimizer # noqa
from mmdeploy.core import RewriterContext
from mmdeploy.apis.onnx.passes import optimize_onnx

# load you model here
model = create_model()

# export with ONNX Optimizer
x = create_dummy_input()
with RewriterContext({}, onnx_custom_passes=optimize_onnx):
    torch.onnx.export(model, x, output_path)
```

The model would be optimized after export.

You can also define your own optimizer:

```python
# create the optimize callback
def _optimize_onnx(graph, params_dict, torch_out):
    from mmdeploy.backend.torchscript import ts_optimizer
    ts_optimizer.onnx._jit_pass_onnx_peephole(graph)
    return graph, params_dict, torch_out

with RewriterContext({}, onnx_custom_passes=_optimize_onnx):
    # export your model
```
