#include "mmdeploy_Context.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/common.h"
#include "mmdeploy/apis/c/mmdeploy/executor.h"
#include "mmdeploy/apis/c/mmdeploy/model.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_Context_create(JNIEnv *env, jobject) {
  mmdeploy_context_t context{};
  mmdeploy_context_create(&context);
  return (jlong)context;
}

jint Java_mmdeploy_Context_add(JNIEnv *env, jobject, jlong context_, jint contextType, jstring name,
                               jlong handle) {
  auto object_name = env->GetStringUTFChars(name, nullptr);
  if ((int)contextType == MMDEPLOY_TYPE_SCHEDULER) {
    mmdeploy_context_add((mmdeploy_context_t)context_, (mmdeploy_context_type_t)contextType,
                         object_name, (mmdeploy_scheduler_t)handle);
  } else if ((int)contextType == MMDEPLOY_TYPE_MODEL) {
    mmdeploy_context_add((mmdeploy_context_t)context_, (mmdeploy_context_type_t)contextType,
                         object_name, (mmdeploy_model_t)handle);
  } else if ((int)contextType == MMDEPLOY_TYPE_DEVICE) {
    mmdeploy_context_add((mmdeploy_context_t)context_, (mmdeploy_context_type_t)contextType,
                         nullptr, (mmdeploy_device_t)handle);
  } else if ((int)contextType == MMDEPLOY_TYPE_PROFILER) {
    mmdeploy_context_add((mmdeploy_context_t)context_, (mmdeploy_context_type_t)contextType,
                         nullptr, (mmdeploy_profiler_t)handle);
  } else {
    MMDEPLOY_ERROR("wrong context type, got {}", (int)contextType);
    return MMDEPLOY_E_NOT_SUPPORTED;
  }
  env->ReleaseStringUTFChars(name, object_name);
  return 0;
}

void Java_mmdeploy_Context_destroy(JNIEnv *, jobject, jlong context_) {
  MMDEPLOY_DEBUG("Java_mmdeploy_Context_destroy");
  mmdeploy_context_destroy((mmdeploy_context_t)context_);
}
