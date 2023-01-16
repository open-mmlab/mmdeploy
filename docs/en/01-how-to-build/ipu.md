# build for IPU

## 1、enable poplar from docker
pull poplar docker image from dockerhub:
$ docker pull graphcore/poplar
image description：[poplar-docker](https://hub.docker.com/r/graphcore/poplar)

create docker container with following command prefix:
$ docker run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK --ipc=host --device=/dev/ipu0:/dev/ipu0 --device=/dev/ipu0_ex:/dev/ipu0_ex --device=/dev/ipu0_mem:/dev/ipu0_mem 

## 2、build mmdeploy SDK and demo
1. install c++ dependencies
$  apt-get install libopencv-dev libboost-all-dev gcc-7 g++-7 cmake
2. install conda
$  download [anaconda](https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh)
   bash Anaconda3-2022.10-Linux-x86_64.sh
   source ~/.bashrc
3. create python3.8 env
$ conda create --name ipu python=3.8
4. build mmdeploy
$ cd to mmdeploy dir
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_BUILD_SDK=ON -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_TARGET_DEVICES=cpu -DMMDEPLOY_TARGET_BACKENDS=ipu -DMMDEPLOY_BUILD_TEST=ON
$ make -j$(nproc)

## 3、install ipu converter
to fetch detailed ipu converter install guide, plese refer to https://docs.graphcore.ai/projects/poprt-user-guide/en/latest/installation.html
