#include "mmdeploy_Segmentor.h"

#include <numeric>

#include "mmdeploy/apis/c/segmentor.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_Segmentor_create(JNIEnv *env, jobject, jstring modelPath, jstring deviceName,
                                     jint device_id) {
  auto model_path = env->GetStringUTFChars(modelPath, nullptr);
  auto device_name = env->GetStringUTFChars(deviceName, nullptr);
  mm_handle_t segmentor{};
  auto ec = mmdeploy_segmentor_create_by_path(model_path, device_name, (int)device_id, &segmentor);
  env->ReleaseStringUTFChars(modelPath, model_path);
  env->ReleaseStringUTFChars(deviceName, device_name);
  if (ec) {
    MMDEPLOY_ERROR("failed to create segmentor, code = {}", ec);
  }
  return (jlong)segmentor;
}

void Java_mmdeploy_Segmentor_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_INFO("Java_mmdeploy_Segmentor_destroy");
  mmdeploy_segmentor_destroy((mm_handle_t)handle);
}

jobjectArray Java_mmdeploy_Segmentor_apply(JNIEnv *env, jobject thiz, jlong handle,
                                           jobjectArray images) {
  return With(env, images, [&](const mm_mat_t imgs[], int size) {
    mm_segment_t *results{};
    auto ec = mmdeploy_segmentor_apply((mm_handle_t)handle, imgs, size, &results);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply segmentor, code = {}", ec);
    }

    auto result_cls = env->FindClass("mmdeploy/Segmentor$Result");
    auto result_ctor = env->GetMethodID(result_cls, "<init>", "(III[I)V");
    auto array = env->NewObjectArray(size, result_cls, nullptr);

    for (int i = 0; i < size; ++i) {
      int *mask = results[i].mask;
      jintArray jmask = env->NewIntArray(results[i].height * results[i].width);
      env->SetIntArrayRegion(jmask, 0, results[i].width * results[i].height, (const jint *)mask);
      auto res = env->NewObject(result_cls, result_ctor, (jint)results[i].height,
                                (jint)results[i].width, (jint)results[i].classes, jmask);
      env->SetObjectArrayElement(array, i, res);
    }
    mmdeploy_segmentor_release_result(results, size);
    return array;
  });
}
