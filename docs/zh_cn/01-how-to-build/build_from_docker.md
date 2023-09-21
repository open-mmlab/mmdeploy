# 使用 Docker 镜像

本文简述如何使用[Docker](https://docs.docker.com/get-docker/)安装mmdeploy

## 获取镜像

为了方便用户，mmdeploy在[Docker Hub](https://hub.docker.com/r/openmmlab/mmdeploy)上提供了多个版本的镜像，例如对于`mmdeploy==1.2.0`，
其镜像标签为`openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy1.2.0`，而最新的镜像标签为`openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy`。
镜像相关规格信息如下表所示：

|    Item     |   Version   |
| :---------: | :---------: |
|     OS      | Ubuntu20.04 |
|    CUDA     |    11.8     |
|    CUDNN    |     8.9     |
|   Python    |   3.8.10    |
|    Torch    |    2.0.0    |
| TorchVision |   0.15.0    |
| TorchScript |    2.0.0    |
|  TensorRT   |   8.6.1.6   |
| ONNXRuntime |   1.15.1    |
|  OpenVINO   |  2022.3.0   |
|    ncnn     |  20230816   |
|   openppl   |    0.8.1    |

用户可选择一个[镜像](https://hub.docker.com/r/openmmlab/mmdeploy/tags)并运行`docker pull`拉取镜像到本地：

```shell
export TAG=openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy
docker pull $TAG
```

## 构建镜像(可选)

如果已提供的镜像无法满足要求，用户可修改`docker/Release/Dockerfile`并在本地构建镜像。其中，构建参数`MMDEPLOY_VERSION`可以是[mmdeploy](https://github.com/open-mmlab/mmdeploy)项目的一个[标签](https://github.com/open-mmlab/mmdeploy/tags)或者分支。

```shell
export MMDEPLOY_VERSION=main
export TAG=mmdeploy-${MMDEPLOY_VERSION}
docker build docker/Release/ -t ${TAG} --build-arg MMDEPLOY_VERSION=${MMDEPLOY_VERSION}
```

## 运行 docker 容器

当拉取或构建 docker 镜像后，用户可使用 `docker run` 启动 docker 服务：

```shell
export TAG=openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy
docker run --gpus=all -it --rm $TAG
```

## 常见问答

1. CUDA error: the provided PTX was compiled with an unsupported toolchain:

   如 [这里](https://forums.developer.nvidia.com/t/cuda-error-the-provided-ptx-was-compiled-with-an-unsupported-toolchain/185754)所说，更新 GPU 的驱动到您的GPU能使用的最新版本。

2. docker: Error response from daemon: could not select device driver "" with capabilities: [gpu].

   ```shell
   # Add the package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
