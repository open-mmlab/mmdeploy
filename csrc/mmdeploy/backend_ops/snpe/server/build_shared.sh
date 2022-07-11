export ANDROID_NDK=/home/PJLAB/konghuanjun/Downloads/android-ndk-r17c
cmake .. \
 -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI=arm64-v8a \
 -DANDROID_PLATFORM=android-26 \
 -DANDROID_STL=c++_shared \
 -DCMAKE_BUILD_TYPE=Release \
 -Dabsl_DIR=/tmp/android_grpc_install_shared/lib/cmake/absl \
 -DProtobuf_DIR=/tmp/android_grpc_install_shared/lib/cmake/protobuf \
 -DgRPC_DIR=/tmp/android_grpc_install_shared/lib/cmake/grpc
