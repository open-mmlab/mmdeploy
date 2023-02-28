#include "mmdeploy_Profiler.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/common.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_Profiler_create(JNIEnv *env, jobject, jstring path) {
  auto profiler_path = env->GetStringUTFChars(path, nullptr);
  mmdeploy_profiler_t profiler{};
  auto ec = mmdeploy_profiler_create(profiler_path, &profiler);
  env->ReleaseStringUTFChars(path, profiler_path);
  if (ec) {
    MMDEPLOY_ERROR("failed to create profiler, code = {}", ec);
    return -1;
  }
  return (jlong)profiler;
}

void Java_mmdeploy_Profiler_destroy(JNIEnv *, jobject, jlong profiler_) {
  MMDEPLOY_DEBUG("Java_mmdeploy_Profiler_destroy");
  mmdeploy_profiler_destroy((mmdeploy_profiler_t)profiler_);
}
