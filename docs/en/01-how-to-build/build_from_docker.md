# Use Docker Image

We provide two dockerfiles for CPU and GPU respectively. For CPU users, we install MMDeploy with ONNXRuntime, ncnn and OpenVINO backends. For GPU users, we install MMDeploy with TensorRT backend. Besides, users can install mmdeploy with different versions when building the docker image.

## Build docker image

For CPU users, we can build the docker image with the latest MMDeploy through:

```
cd mmdeploy
docker build docker/CPU/ -t mmdeploy:master-cpu
```

For GPU users, we can build the docker image with the latest MMDeploy through:

```
cd mmdeploy
docker build docker/GPU/ -t mmdeploy:master-gpu
```

For installing MMDeploy with a specific version, we can append `--build-arg VERSION=${VERSION}` to build command. GPU for example:

```
cd mmdeploy
docker build docker/GPU/ -t mmdeploy:0.1.0 --build-arg  VERSION=0.1.0
```

For installing libs with the aliyun source, we can append `--build-arg USE_SRC_INSIDE=${USE_SRC_INSIDE}` to build command.

```
# GPU for example
cd mmdeploy
docker build docker/GPU/ -t mmdeploy:inside --build-arg  USE_SRC_INSIDE=true

# CPU for example
cd mmdeploy
docker build docker/CPU/ -t mmdeploy:inside --build-arg  USE_SRC_INSIDE=true
```

## Run docker container

After building the docker image succeed, we can use `docker run` to launch the docker service. GPU docker image for example:

```
docker run --gpus all -it mmdeploy:master-gpu
```

## FAQs

1. CUDA error: the provided PTX was compiled with an unsupported toolchain:

   As described [here](https://forums.developer.nvidia.com/t/cuda-error-the-provided-ptx-was-compiled-with-an-unsupported-toolchain/185754), update the GPU driver to the latest one for your GPU.

2. docker: Error response from daemon: could not select device driver "" with capabilities: \[gpu\].

   ```
   # Add the package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
