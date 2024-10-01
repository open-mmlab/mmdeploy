# 使用Jetson Docker镜像

本文档指导如何在Jetson上通过[Docker](https://docs.docker.com/get-docker/)安装mmdeploy。

## 获取预构建的docker镜像

MMDeploy为用户在[Docker Hub](https://hub.docker.com/r/openmmlab/mmdeploy)上提供了预构建的docker镜像。这些docker镜像基于最新和已发布的版本构建。我们发布了两个版本的docker镜像，分别为Jetpack=5.1和Jetpack=4.6.1。例如，标签为`openmmlab/mmdeploy_jetpack5:v1`的镜像是为Jetpack5.1构建的，而标签为`openmmlab/mmdeploy_jetpack4.6.1:v1`的镜像则是为Jetpack 4.6.1构建的。Docker镜像的规格如下所示。

- jetpack5.1

|    项目    |   版本    |
| :--------: | :------: |
| Jetpack    |   5.1    |
| Python     |  3.8.10  |
| Torch      |   2.0.0  |
| TorchVision|  0.15.0  |

- jetpack4.6.1

|    项目    |   版本    |
| :--------: | :------: |
| Jetpack    |   4.6.1  |
| Python     |  3.8.10  |
| Torch      |  1.10.0  |
| TorchVision|  0.11.0  |

- jetpack 5.1
```shell
export TAG=openmmlab/mmdeploy_jetpack5:v1
docker pull $TAG
```
- jetpack 4.6.1
```shell
export TAG=openmmlab/mmdeploy_jetpack4.6:v1
docker pull $TAG
```

## 构建docker镜像（可选）
如果预构建的docker镜像不符合您的要求，您可以通过运行以下脚本来构建自己的镜像。docker文件分别为docker/jetson/jetpack5/Dockerfile和docker/jetson/jetpack4.6/Dockerfile，

```shell
docker build docker/jetson/jetpack5 -t openmmlab/mmdeploy_jetpack5:v1 .
//
docker build docker/jetson/jetpack4.6 -t openmmlab/mmdeploy_jetpack4.6:v1 .
```

## 运行docker容器
拉取或构建docker镜像后，您可以使用docker run来启动docker服务：

```shell
docker run -it --rm --runtime nvidia --network host openmmlab/mmdeploy_jetpack5:v1
//
docker run -it --rm --runtime nvidia --network host openmmlab/mmdeploy_jetpack4.6:v1
```

## 故障排除
如果您使用的是jetpack5，可能需要解决一些问题。

1. OpenCV问题
如果您发现import cv2时出错，找不到libpng15.so
```shell
  ln -s /usr/local/lib/python3.x/dist-packages/opencv-python.libs/* /usr/lib
```
2. mmdetection问题
如果您发现安装了mmdetection，但无法导入mmdet。您可以使用以下命令来安装
```shell
  python3 -m pip install --user -e .
```
3. Jetson分布式问题(已在PR中重写)
如果您按照Jetson.md中的方法转换模型，您可能会发现torch.distributed没有ReduceOp属性。我只是提出了问题并做了一个简单的补丁，在./mmdeploy/tools/下添加jetson_patch.py文件
```python
import torch.distributed
if not torch.distributed.is_available():
  torch.distributed.ReduceOp = lambda: None
```
并在您需要的文件开头导入jetson_patch。我知道这并不优雅，但它确实有效...(适用于Jetson AGX Orin)

4. Jetpack5.1 PyTorch2.0有一些问题
> 如果您直接使用我们的预编译docker镜像或者使用dockerfile进行构建镜像，这个问题已在dockerfile中解决

  我们需要修改 torch.onnx._run_symbolic_method 这个函数。
  **从**
```python
def _run_symbolic_method(g, op_name, symbolic_fn, args):
    r"""
    This trampoline function gets invoked for every symbolic method
    call from C++.
    """
    try:
        return symbolic_fn(g, *args)
    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch
        # to symbolic_fn.  Otherwise, the backtrace will have the clues
        # you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name),)
        raise
```
  **到**
```python
@_beartype.beartype
def _run_symbolic_method(g, op_name, symbolic_fn, args):
    r"""
    This trampoline function gets invoked for every symbolic method
    call from C++.
    """
    try:
        graph_context = jit_utils.GraphContext(
            graph=g,
            block=g.block(),
            opset=GLOBALS.export_onnx_opset_version,
            original_node=None,  # type: ignore[arg-type]
            params_dict=_params_dict,
            env={},
        )
        return symbolic_fn(graph_context, *args)
    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch
        # to symbolic_fn.  Otherwise, the backtrace will have the clues
        # you need.
        e.args = (f"{e.args[0]} (occurred when translating {op_name})",)
        raise
  ```
最后您就可以开心的使用这个镜像了:)
