# 支持 IPU

## 一、配置poplar环境
从dockerhub拉取poplar的docker镜像: 
$ docker pull graphcore/poplar
镜像描述详见：[poplar-docker](https://hub.docker.com/r/graphcore/poplar)

用以下docker命令前缀创建容器
$ docker run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK --ipc=host --device=/dev/ipu0:/dev/ipu0 --device=/dev/ipu0_ex:/dev/ipu0_ex --device=/dev/ipu0_mem:/dev/ipu0_mem 

## 二、编译 mmdeploy SDK 和 demo
1. 安装c++相关依赖
$  apt-get install libopencv-dev libboost-all-dev gcc-7 g++-7 cmake
2. 安装conda
$  downlaod [anaconda](https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh)
   bash Anaconda3-2022.10-Linux-x86_64.sh
   source ~/.bashrc
3. 创建python3.8环境
$ conda create --name ipu python=3.8
4. 编译mmdeploy
$ cd to mmdeploy dir
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_BUILD_SDK=ON -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_TARGET_DEVICES=cpu -DMMDEPLOY_TARGET_BACKENDS=ipu -DMMDEPLOY_BUILD_TEST=ON
$ make -j$(nproc)

## 三、安装ipu converter
目前ipu的converter暂时没有开源，如果您对ipu试用感兴趣，请联系c600-support@graphcore.ai