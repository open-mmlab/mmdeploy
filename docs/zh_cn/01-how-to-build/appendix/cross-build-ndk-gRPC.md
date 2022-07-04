# Linux 交叉编译 android gRPC

mmdeploy 已提供 prebuilt snpe inference server，如果你想自己编译、或需要对 .proto 接口做修改，可参考此文档。

注意 gRPC 官方文档并没有对 NDK 的完整支持。

## 环境说明

| 项目 | 版本 | 备注 |
| ------ | ----- | ------ |
| snpe | 1.63.0.3523 | - |
| host OS | ubuntu18.04 | snpe1.63.0 文档指定版本 |
| NDK | r17c | snpe1.63.0 文档指定版本 |
| gRPC | commit 6f698b5 | - |
| 硬件设备 | 红米 K40 | 需要 qcom 芯片 |

## NDK 交叉编译 gRPC
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
 -DANDROID_STL=c++_static \
 -DRUN_HAVE_STD_REGEX=0 \
 -DRUN_HAVE_POSIX_REGEX=0 \
 -DRUN_HAVE_STEADY_CLOCK=0 \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=/tmp/android_grpc_install
$ make -j
$ make install
```

3. install 结束后，`/tmp/android_grpc_install` 应有完整的安装文件
```bash
$ cd /tmp/android_grpc_install
$ tree -L 1
.
├── bin
├── include
├── lib
└── share
```

## 测试 gRPC
1. 编译 gRPC 自带的 helloworld
```bash
$ cd /path/to/grpc/examples/cpp/helloworld/
$ mkdir cmake/build_aarch64 -p && pushd cmake/build_aarch64

$ cmake ../.. \
 -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI=arm64-v8a \
 -DANDROID_PLATFORM=android-26 \
 -DANDROID_STL=c++_static \
 -DRUN_HAVE_STD_REGEX=0 \
 -DRUN_HAVE_POSIX_REGEX=0 \
 -DRUN_HAVE_STEADY_CLOCK=0 \
 -DCMAKE_BUILD_TYPE=Release \
 -Dabsl_DIR=/tmp/android_grpc_install/lib/cmake/absl \
 -DProtobuf_DIR=/tmp/android_grpc_install/lib/cmake/protobuf \
 -DgRPC_DIR=/tmp/android_grpc_install/lib/cmake/grpc
$ make -j
$ ls greeter*
greeter_async_client   greeter_async_server     greeter_callback_server  greeter_server
greeter_async_client2  greeter_callback_client  greeter_client
```
2. 打卡手机的 adb 调试模式，push 编译结果到 `/data/local/tmp` 目录
```bash
$ adb push greeter* /data/local/tmp
```
3. `adb shell` 进手机，执行 client/server
```bash
/data/local/tmp $ ./greeter_client                                        
Greeter received: Hello world
```

## 参考文档

* gRPC cross build script https://raw.githubusercontent.com/grpc/grpc/master/test/distrib/cpp/run_distrib_test_cmake_aarch64_cross.sh
* stackoverflow https://stackoverflow.com/questions/54052229/build-grpc-c-for-android-using-ndk-arm-linux-androideabi-clang-compiler
