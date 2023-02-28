#include "mmdeploy_Model.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/model.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_Model_create(JNIEnv *env, jobject, jstring path) {
  auto model_path = env->GetStringUTFChars(path, nullptr);
  mmdeploy_model_t model{};
  auto ec = mmdeploy_model_create_by_path(model_path, &model);
  env->ReleaseStringUTFChars(path, model_path);
  if (ec) {
    MMDEPLOY_ERROR("failed to create model, code = {}", ec);
    return -1;
  }
  return (jlong)model;
}

void Java_mmdeploy_Model_destroy(JNIEnv *, jobject, jlong model_) {
  MMDEPLOY_DEBUG("Java_mmdeploy_Model_destroy");
  mmdeploy_model_destroy((mmdeploy_model_t)model_);
}
