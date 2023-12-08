# Use Jetson Docker Image

This document guides how to install mmdeploy with [Docker](https://docs.docker.com/get-docker/) on Jetson.

## Get prebuilt docker images

MMDeploy provides prebuilt docker images for the convenience of its users on [Docker Hub](https://hub.docker.com/r/openmmlab/mmdeploy). The docker images are built on
the latest and released versions. We release two docker version, for Jetpack=5.1 and Jetpack=4.6.1
For instance, the image with tag `openmmlab/mmdeploy_jetpack5:v1` is built for Jetpack5.1 and the image with tag `openmmlab/mmdeploy_jetpack4.6.1:v1` is build for Jetpack 4.6.1.
The specifications of the Docker Images are shown below.

- jetpack5.1

|    Item     |   Version   |
| :---------: | :---------: |
| Jetpack     |     5.1     |
|   Python    |   3.8.10    |
|    Torch    |    2.0.0    |
| TorchVision |   0.15.0    |

- jetpack4.6.1

|    Item     |   Version   |
| :---------: | :---------: |
|   Jetpack   |   4.6.1     |
|   Python    |   3.8.10    |
|    Torch    |   1.10.0    |
| TorchVision |   0.11.0    |

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
## Build docker images (optional)

If the prebuilt docker images do not meet your requirements,
then you can build your own image by running the following script.
The docker file is `docker/jetson/jetpack5/Dockerfile` and `docker/jetson/jetpack4.6/Dockerfile`,

```shell
sudo docker build docker/jetson/jetpack5 -t openmmlab/mmdeploy_jetpack5:v1 .
//
sudo docker build docker/jetson/jetpack4.6 -t openmmlab/mmdeploy_jetpack4.6:v1 .
```

## Run docker container

After pulling or building the docker image, you can use `docker run` to launch the docker service:

```shell
docker run -it --rm --runtime nvidia --network host openmmlab/mmdeploy_jetpack5:v1
//
docker run -it --rm --runtime nvidia --network host openmmlab/mmdeploy_jetpack4.6:v1
```

## TroubleShooting
If you using the jetpack5, it has some question need to solve.
1. OpenCV problem
  if you find import cv2 wrong, can't find the libpng15.so
```shell
  ln -s /usr/local/lib/python3.x/dist-packages/opencv-python.libs/* /usr/lib
```

2. mmdetection problem
  if you find installed the mmdetection, but import the mmdet failed. you should use this to install
```shell
  python3 -m pip install --user -e .
```

3. Jetson No distributed problem
  if you convert the model like [Jetson.md](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/01-how-to-build/jetsons.md)
  you may find torch.distributed has no attribute ReduceOp.
  I just issue and make a simple patch, add file jetson_patch.py on ./mmdeploy/tools/
```python
import torch.distributed
if not torch.distributed.is_available():
  torch.distributed.ReduceOp = lambda: None
```
  and import jetson_patch at the beginning which file you want.
  I know is not quietly ellegant, but it works well...(for Jetson AGX Orin)
4. Jetpack with PyTorch 2.0 has some issue
  we need to modify torch.onnx._run_symbolic_method 
  **from**
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
  **to**
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
Finally we can use Jetpack5.1 && MMDeploy happily:)
