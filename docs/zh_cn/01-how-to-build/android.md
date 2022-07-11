# Android 下构建方式

- [Android 下构建方式](#android-下构建方式)
  - [源码安装](#源码安装)
    - [安装构建和编译工具链](#安装构建和编译工具链)
    - [安装依赖包](#安装依赖包)
      - [安装 MMDeploy SDK 依赖](#安装-mmdeploy-sdk-依赖)
    - [编译 MMDeploy](#编译-mmdeploy)
      - [编译选项说明](#编译选项说明)
      - [编译 SDK](#编译-sdk)
      - [编译 Demo](#编译-demo)

______________________________________________________________________

MMDeploy 为 android 平台提供交叉编译的构建方式.

MMDeploy converter 部分在 linux 平台上执行,SDK 部分在 android 平台上执行.

MMDeploy 的交叉编译分为两步:

1. 在 linux 平台上构建 MMDeploy converter. 请根据 [How to build linux](./linux-x86_64.md) 进行构建.

2. 使用 android 工具链构建 MMDeploy SDK.

本文档仅提供在 linux 平台上使用 android 工具链进行交叉编译构建 MMDeploy SDK 的方法.

## 源码安装

### 安装构建和编译工具链

- cmake

  **保证 cmake的版本 >= 3.14.0**. 如果不是,可以参考以下命令安装 3.20.0 版本. 如要获取其他版本,请参考 [这里](https://cmake.org/install)

  ```bash
  wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.tar.gz
  tar -xzvf cmake-3.20.0-linux-x86_64.tar.gz
  sudo ln -sf $(pwd)/cmake-3.20.0-linux-x86_64/bin/* /usr/bin/
  ```

- ANDROID NDK 19+

  **保证 android ndk 的版本 >= 19.0**. 如果不是,可以参考以下命令安装 r23b 版本. 如要获取其他版本,请参考 [这里](https://developer.android.com/ndk/downloads)

  ```bash
  wget https://dl.google.com/android/repository/android-ndk-r23b-linux.zip
  unzip android-ndk-r23b-linux.zip
  cd android-ndk-r23b
  export NDK_PATH=${PWD}
  ```

### 安装依赖包

#### 安装 MMDeploy SDK 依赖

如果您只对模型转换感兴趣,那么可以跳过本章节.

<table>
<thead>
  <tr>
    <th>名称 </th>
    <th>安装方式 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>OpenCV<br>(>=3.0) </td>
    <td>
<pre><code>
export OPENCV_VERSION=4.5.4
wget https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/opencv-${OPENCV_VERSION}-android-sdk.zip
unzip opencv-${OPENCV_VERSION}-android-sdk.zip
export OPENCV_ANDROID_SDK_DIR=${PWD}/OpenCV-android-sdk
</code></pre>
    </td>

</tr>
  <tr>
    <td>ncnn </td>
    <td>ncnn 是支持 android 平台的高效神经网络推理计算框架</br>
  <b> 目前, MMDeploy 支持 ncnn 的 20220216 版本, 且必须使用<code>git clone</code> 下载源码的方式安装</b><br>
<pre><code>
git clone -b 20220216 https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
export NCNN_DIR=${PWD}
mkdir -p build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-30 -DNCNN_VULKAN=ON -DNCNN_DISABLE_EXCEPTION=OFF -DNCNN_DISABLE_RTTI=OFF ..
make install
</code></pre>
   </td>
  </tr>
    <tr>
  <td>OpenJDK </td>
  <td>编译Java API之前需要先准备OpenJDK开发环境</br>
  请参考 <a href='https://github.com/open-mmlab/mmdeploy/blob/master/csrc/mmdeploy/apis/java/README.md'> Java API 编译 </a> 进行构建.
  </td>
  </tr>
</tbody>
</table>

### 编译 MMDeploy

#### 编译选项说明

<table>
<thead>
  <tr>
    <th>编译选项</th>
    <th>取值范围</th>
    <th>缺省值</th>
    <th>说明</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MMDEPLOY_BUILD_SDK</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>MMDeploy SDK 编译开关</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_SDK_JAVA_API</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>MMDeploy SDK Java API的编译开关</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_TEST</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>MMDeploy SDK 的单元测试程序编译开关</td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_DEVICES</td>
    <td>{"cpu"}</td>
    <td>cpu</td>
    <td>设置目标设备. <br>如果您想使用 ncnn 的 vulkan 加速功能, 您仍旧需要在这里填写<code>{"cpu"}</code>参数. 这是因为 vulkan 加速仅用于 ncnn 网络计算内部,推理过程的其他部分仍旧使用 cpu 计算.</td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_BACKENDS</td>
    <td>{"ncnn"}</td>
    <td>N/A</td>
    <td>设置推理后端. <br><b>默认情况下,SDK不设置任何后端, 因为它与应用场景高度相关.</b><br> Android 端目前只支持ncnn一个后端 <br>
    构建时,几乎每个后端,都需传入一些路径变量,用来查找依赖包. <br>
    1. <b>ncnn</b>: 表示 ncnn. 需要设置<code>ncnn_DIR</code>.
<pre><code>-Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn</code></pre>
   </td>
  </tr>
  <tr>
    <td>MMDEPLOY_CODEBASES</td>
    <td>{"mmcls", "mmdet", "mmseg", "mmedit", "mmocr", "all"}</td>
    <td>N/A</td>
    <td>用来设置 SDK 后处理组件,加载 OpenMMLab 算法仓库的后处理功能. 已支持的算法仓库有'mmcls','mmdet','mmedit','mmseg'和'mmocr'. 如果选择多个codebase,中间使用分号隔开. 比如, 'mmcls', 'mmdet', 'mmedit', 'mmseg', 'mmocr'. 也可以通过 <code>all</code> 的方式, 加载所有codebase, 即 <code>-DMMDEPLOY_CODEBASES=all.</code></td>
  </tr>
  <tr>
    <td>MMDEPLOY_SHARED_LIBS</td>
    <td>{ON, OFF}</td>
    <td>ON</td>
    <td>MMDeploy SDK 的动态库的编译开关.设置 OFF 时, 编译静态库. 目前 android 端 SDK 仅支持静态库加载, 后续会进行对动态库加载的支持.</td>
  </tr>
</tbody>
</table>

#### 编译 SDK

下文展示构建SDK的样例，用 ncnn 作为推理引擎。

- cpu + ncnn
  ```Bash
  cd ${MMDEPLOY_DIR}
  mkdir -p build && cd build
  cmake .. \
      -DMMDEPLOY_BUILD_SDK=ON \
      -DMMDEPLOY_BUILD_SDK_JAVA_API=ON \
      -DOpenCV_DIR=${OPENCV_ANDROID_SDK_DIR}/sdk/native/jni/abi-arm64-v8a \
      -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn \
      -DMMDEPLOY_TARGET_BACKENDS=ncnn \
      -DMMDEPLOY_CODEBASES=all \
      -DMMDEPLOY_SHARED_LIBS=OFF \
      -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-30 \
      -DANDROID_CPP_FEATURES="rtti exceptions"

  make -j$(nproc) && make install
  ```

#### 编译 Demo

```Bash
cd ${MMDEPLOY_DIR}/build/install/example
mkdir -p build && cd build
cmake .. \
      -DOpenCV_DIR=${OPENCV_ANDROID_SDK_DIR}/sdk/native/jni/abi-arm64-v8a \
      -Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn \
      -DMMDeploy_DIR=${MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy \
      -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-30
make -j$(nproc)
```
