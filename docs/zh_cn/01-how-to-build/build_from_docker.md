# 使用 Docker 镜像

我们分别为 CPU 和 GPU 提供了两个 dockerfile。对于 CPU 用户，我们对接 ONNXRuntime、ncnn 和 OpenVINO 后端安装 MMDeploy。对于 GPU 用户，我们安装带有 TensorRT 后端的 MMDeploy。此外，用户可以在构建 docker 镜像时安装不同版本的 mmdeploy。

## 构建镜像

对于 CPU 用户，我们可以通过以下方式使用最新的 MMDeploy 构建 docker 镜像：

```
cd mmdeploy
docker build docker/CPU/ -t mmdeploy:master-cpu
```

对于 GPU 用户，我们可以通过以下方式使用最新的 MMDeploy 构建 docker 镜像：

```
cd mmdeploy
docker build docker/GPU/ -t mmdeploy:master-gpu
```

要安装具有特定版本的 MMDeploy，我们可以将 `--build-arg VERSION=${VERSION}` 附加到构建命令中。以 GPU 为例：

```
cd mmdeploy
docker build docker/GPU/ -t mmdeploy:0.1.0 --build-arg  VERSION=0.1.0
```

要切换成阿里源安装依赖，我们可以将 `--build-arg USE_SRC_INSIDE=${USE_SRC_INSIDE}` 附加到构建命令中。

```
# 以 GPU 为例
cd mmdeploy
docker build docker/GPU/ -t mmdeploy:inside --build-arg  USE_SRC_INSIDE=true

# 以 CPU 为例
cd mmdeploy
docker build docker/CPU/ -t mmdeploy:inside --build-arg  USE_SRC_INSIDE=true
```

## 运行 docker 容器

构建 docker 镜像成功后，我们可以使用 `docker run` 启动 docker 服务。 GPU 镜像为例：

```
docker run --gpus all -it -p 8080:8081 mmdeploy:master-gpu
```

## 常见问答

1. CUDA error: the provided PTX was compiled with an unsupported toolchain:

   如 [这里](https://forums.developer.nvidia.com/t/cuda-error-the-provided-ptx-was-compiled-with-an-unsupported-toolchain/185754)所说，更新 GPU 的驱动到您的GPU能使用的最新版本。

2. docker: Error response from daemon: could not select device driver "" with capabilities: \[\[gpu\]\].

   ```
   # Add the package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
