## Docker usage

We provide two dockerfiles for CPU and GPU respectively. For CPU users, we install MMDeploy with ONNXRuntime, ncnn and OpenVINO backends. For GPU users, we install MMDeploy with TensorRT backend. Besides, users can install mmdeploy with different versions when building the docker image.

### Build docker image

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

### Run docker container

After building the docker image succeed, we can use `docker run` to launch the docker service. GPU docker image for example:
```
docker run --gpus all -it -p 8080:8081 mmdeploy:master-gpu
```

### AFQs

1. CUDA error: the provided PTX was compiled with an unsupported toolchain:

    As described [here](https://forums.developer.nvidia.com/t/cuda-error-the-provided-ptx-was-compiled-with-an-unsupported-toolchain/185754), update the GPU driver to the latest one for your GPU.
