# Use Docker Image

This document guides how to install mmdeploy with [Docker](https://docs.docker.com/get-docker/).

## Get prebuilt docker images

MMDeploy provides prebuilt docker images for the convenience of its users on [Docker Hub](https://hub.docker.com/r/openmmlab/mmdeploy). The docker images are built on
the latest and released versions. For instance, the image with tag `openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy` is built on the latest mmdeploy and the image with tag `openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy1.2.0` is for `mmdeploy==1.2.0`.
The specifications of the Docker Image are shown below.

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

You can select a [tag](https://hub.docker.com/r/openmmlab/mmdeploy/tags) and run `docker pull` to get the docker image:

```shell
export TAG=openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy
docker pull $TAG
```

## Build docker images (optional)

If the prebuilt docker images do not meet your requirements,
then you can build your own image by running the following script.
The docker file is `docker/Release/Dockerfile`and its building argument is `MMDEPLOY_VERSION`,
which can be a [tag](https://github.com/open-mmlab/mmdeploy/tags) or a branch from [mmdeploy](https://github.com/open-mmlab/mmdeploy).

```shell
export MMDEPLOY_VERSION=main
export TAG=mmdeploy-${MMDEPLOY_VERSION}
docker build docker/Release/ -t ${TAG} --build-arg MMDEPLOY_VERSION=${MMDEPLOY_VERSION}
```

## Run docker container

After pulling or building the docker image, you can use `docker run` to launch the docker service:

```shell
export TAG=openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy
docker run --gpus=all -it --rm $TAG
```

## FAQs

1. CUDA error: the provided PTX was compiled with an unsupported toolchain:

   As described [here](https://forums.developer.nvidia.com/t/cuda-error-the-provided-ptx-was-compiled-with-an-unsupported-toolchain/185754), update the GPU driver to the latest one for your GPU.

2. docker: Error response from daemon: could not select device driver "" with capabilities: [gpu].

   ```shell
   # Add the package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
