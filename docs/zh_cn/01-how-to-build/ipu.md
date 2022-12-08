# 支持 IPU

## 一、配置poplar环境
从dockerhub拉取poplar的docker镜像: 
$ docker pull graphcore/poplar
镜像描述详见：[poplar-docker](https://hub.docker.com/r/graphcore/poplar)

## 二、编译 mmdeploy SDK 和 demo
1.安装opencv>=3.0
$  apt-get install libopencv-dev
2.安装cmake>=3.14
$  apt-get install cmake
3.安装boost
$ apt-get install libboost-all-dev
3.安装conda
4.创建python3.8环境
# conda create --name ipu python=3.8
5.编译mmdeploy
$ cmake .. -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_BUILD_SDK=ON -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON -DMMDEPLOY_BUILD_EXAMPLES=ON -DMMDEPLOY_TARGET_DEVICES=cpu -DMMDEPLOY_TARGET_BACKENDS=ipu -DMMDEPLOY_BUILD_TEST=ON
