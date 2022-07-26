# Ubuntu18.04 交叉编译 NDK snpe 推理服务

mmdeploy 已提供预编译包，如果你想自己编译、或需要对 .proto 接口做修改，可参考此文档。

注意 gRPC 官方文档并没有对 NDK 的完整支持。

## 一、环境说明

| 项目     | 版本           | 备注                                  |
| -------- | -------------- | ------------------------------------- |
| snpe     | 1.59           | 1.60 使用 clang-8.0，可能导致兼容问题 |
| host OS  | ubuntu18.04    | snpe1.59 指定版本                     |
| NDK      | r17c           | snpe1.59 指定版本                     |
| gRPC     | commit 6f698b5 | -                                     |
| 硬件设备 | qcom888        | 需要 qcom 芯片                        |

## 二、NDK 交叉编译 gRPC

1. 拉取 gRPC repo,  在 host 上编译出 `protoc` 和 `grpc_cpp_plugin`

```bash
# 安装依赖
$ apt-get update && apt-get install -y libssl-dev
# 编译
$ git clone https://github.com/grpc/grpc --recursive=1 --depth=1
$ mkdir -p cmake/build
$ pushd cmake/build

$ cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DgRPC_INSTALL=ON \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_SSL_PROVIDER=package \
  ../..
# 需要安装到 host 环境
$ make -j
$ sudo make install
```

2. 下载 NDK，交叉编译 android aarch64 所需静态库

```bash
$ wget https://dl.google.com/android/repository/android-ndk-r17c-linux-x86_64.zip
$ unzip android-ndk-r17c-linux-x86_64.zip

# 设置环境变量
$ export ANDROID_NDK=/path/to/android-ndk-r17c

# 编译
$ cd /path/to/grpc
$ mkdir -p cmake/build_aarch64  && pushd cmake/build_aarch64

$ cmake ../.. \
 -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI=arm64-v8a \
 -DANDROID_PLATFORM=android-26 \
 -DANDROID_TOOLCHAIN=clang \
 -DANDROID_STL=c++_shared \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=/tmp/android_grpc_install_shared

$ make -j
$ make install
```

3. 此时 `/tmp/android_grpc_install` 应有完整的安装文件

```bash
$ cd /tmp/android_grpc_install
$ tree -L 1
.
├── bin
├── include
├── lib
└── share
```

## 三、【可跳过】自测 NDK gRPC 是否正常

1. 编译 gRPC 自带的 helloworld

```bash
$ cd /path/to/grpc/examples/cpp/helloworld/
$ mkdir cmake/build_aarch64 -p && pushd cmake/build_aarch64

$ cmake ../.. \
 -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI=arm64-v8a \
 -DANDROID_PLATFORM=android-26 \
 -DANDROID_STL=c++_shared \
 -DANDROID_TOOLCHAIN=clang \
 -DCMAKE_BUILD_TYPE=Release \
 -Dabsl_DIR=/tmp/android_grpc_install_shared/lib/cmake/absl \
 -DProtobuf_DIR=/tmp/android_grpc_install_shared/lib/cmake/protobuf \
 -DgRPC_DIR=/tmp/android_grpc_install_shared/lib/cmake/grpc

$ make -j
$ ls greeter*
greeter_async_client   greeter_async_server     greeter_callback_server  greeter_server
greeter_async_client2  greeter_callback_client  greeter_client
```

2. 打开手机调试模式，push 编译结果到 `/data/local/tmp` 目录

tips：对于国产手机，设置 - 版本号，点击 7 次可进入开发者模式，然后才能打开 USB 调试

```bash
$ adb push greeter* /data/local/tmp
```

3. `adb shell` 进手机，执行 client/server

```bash
/data/local/tmp $ ./greeter_client
Greeter received: Hello world
```

## 四、交叉编译 snpe 推理服务

1. 打开 [snpe tools 官网](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/tools)，下载 1.59 版本。 解压并设置环境变量

**注意 snpe >= 1.60 开始使用 `clang-8.0`，可能导致旧设备与 `libc++_shared.so` 不兼容。**

```bash
$ export SNPE_ROOT=/path/to/snpe-1.59.0.3230
```

2. 打开 mmdeploy  snpe server 目录，使用交叉编译 gRPC 时的选项

```bash
$ cd /path/to/mmdeploy
$ cd service/snpe/server

$ mkdir -p build && cd build
$ export ANDROID_NDK=/path/to/android-ndk-r17c
$ cmake .. \
 -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI=arm64-v8a \
 -DANDROID_PLATFORM=android-26 \
 -DANDROID_STL=c++_shared \
 -DANDROID_TOOLCHAIN=clang \
 -DCMAKE_BUILD_TYPE=Release \
 -Dabsl_DIR=/tmp/android_grpc_install_shared/lib/cmake/absl \
 -DProtobuf_DIR=/tmp/android_grpc_install_shared/lib/cmake/protobuf \
 -DgRPC_DIR=/tmp/android_grpc_install_shared/lib/cmake/grpc

 $ make -j
 $ file inference_server
inference_server: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), dynamically linked, interpreter /system/bin/linker64, BuildID[sha1]=252aa04e2b982681603dacb74b571be2851176d2, with debug_info, not stripped
```

最终可得到 `infernece_server`，`adb push` 到设备上即可执行。

## 五、重新生成 proto 接口

如果改过 `inference.proto`，需要重新生成 .cpp 和 .py 通信接口

```Shell
$ python3 -m pip install grpc_tools --user
$ python3 -m  grpc_tools.protoc -I./ --python_out=./client/ --grpc_python_out=./client/ inference.proto

$ ln -s `which protoc-gen-grpc`
$ protoc --cpp_out=./ --grpc_out=./  --plugin=protoc-gen-grpc=grpc_cpp_plugin  inference.proto
```

## 参考文档

- snpe tutorial https://developer.qualcomm.com/sites/default/files/docs/snpe/cplus_plus_tutorial.html
- gRPC cross build script https://raw.githubusercontent.com/grpc/grpc/master/test/distrib/cpp/run_distrib_test_cmake_aarch64_cross.sh
- stackoverflow https://stackoverflow.com/questions/54052229/build-grpc-c-for-android-using-ndk-arm-linux-androideabi-clang-compiler
