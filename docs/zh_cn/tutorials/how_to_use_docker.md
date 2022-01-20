## Docker的使用

我们分别为 CPU 和 GPU 提供了两个 dockerfile。对于 CPU 用户，我们对接 ONNXRuntime、ncnn 和 OpenVINO 后端安装 MMDeploy。对于 GPU 用户，我们安装带有 TensorRT 后端的 MMDeploy。此外，用户可以在构建 docker 镜像时安装不同版本的 mmdeploy。

### 构建镜像

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

### 运行 docker 容器

构建 docker 镜像成功后，我们可以使用 `docker run` 启动 docker 服务。 GPU 镜像为例：
```
docker run --gpus all -it -p 8080:8081 mmdeploy:master-gpu
```
