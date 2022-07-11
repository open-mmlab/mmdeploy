#include "mmdeploy_Classifier.h"

#include <numeric>

#include "mmdeploy/apis/c/classifier.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_Classifier_create(JNIEnv *env, jobject, jstring modelPath, jstring deviceName,
                                      jint device_id) {
  auto model_path = env->GetStringUTFChars(modelPath, nullptr);
  auto device_name = env->GetStringUTFChars(deviceName, nullptr);
  mm_handle_t classifier{};
  auto ec =
      mmdeploy_classifier_create_by_path(model_path, device_name, (int)device_id, &classifier);
  env->ReleaseStringUTFChars(modelPath, model_path);
  env->ReleaseStringUTFChars(deviceName, device_name);
  if (ec) {
    MMDEPLOY_ERROR("failed to create classifier, code = {}", ec);
  }
  return (jlong)classifier;
}

void Java_mmdeploy_Classifier_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_INFO("Java_mmdeploy_Classifier_destroy");
  mmdeploy_classifier_destroy((mm_handle_t)handle);
}

jobjectArray Java_mmdeploy_Classifier_apply(JNIEnv *env, jobject thiz, jlong handle,
                                            jobjectArray images, jintArray counts) {
  return With(env, images, [&](const mm_mat_t imgs[], int size) {
    mm_class_t *results{};
    int *result_count{};
    auto ec = mmdeploy_classifier_apply((mm_handle_t)handle, imgs, size, &results, &result_count);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply classifier, code = {}", ec);
    }

    auto result_cls = env->FindClass("mmdeploy/Classifier$Result");
    auto result_ctor = env->GetMethodID(result_cls, "<init>", "(IF)V");
    auto total = std::accumulate(result_count, result_count + size, 0);
    auto array = env->NewObjectArray(total, result_cls, nullptr);

    for (int i = 0; i < total; ++i) {
      auto res = env->NewObject(result_cls, result_ctor, (jint)results[i].label_id,
                                (jfloat)results[i].score);
      env->SetObjectArrayElement(array, i, res);
    }
    auto counts_array = env->GetIntArrayElements(counts, nullptr);
    for (int i = 0; i < size; ++i) {
      counts_array[i] = result_count[i];
    }
    env->ReleaseIntArrayElements(counts, counts_array, 0);
    mmdeploy_classifier_release_result(results, result_count, size);
    return array;
  });
}
