#include "mmdeploy_Device.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/common.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_Device_create(JNIEnv *env, jobject, jstring name, jint index) {
  auto device_name = env->GetStringUTFChars(name, nullptr);
  mmdeploy_device_t device{};
  auto ec = mmdeploy_device_create(device_name, (int)index, &device);
  env->ReleaseStringUTFChars(name, device_name);
  if (ec) {
    MMDEPLOY_ERROR("failed to create device, code = {}", ec);
    return -1;
  }
  return (jlong)device;
}

void Java_mmdeploy_Device_destroy(JNIEnv *, jobject, jlong device_) {
  MMDEPLOY_DEBUG("Java_mmdeploy_Device_destroy");
  mmdeploy_device_destroy((mmdeploy_device_t)device_);
}
