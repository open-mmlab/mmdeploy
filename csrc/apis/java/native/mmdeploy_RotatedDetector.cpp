#include "mmdeploy_RotatedDetector.h"

#include <numeric>

#include "apis/c/rotated_detector.h"
#include "apis/java/native/common.h"
#include "core/logger.h"

jlong Java_mmdeploy_RotatedDetector_create(JNIEnv *env, jobject, jstring modelPath,
                                           jstring deviceName, jint device_id) {
  auto model_path = env->GetStringUTFChars(modelPath, nullptr);
  auto device_name = env->GetStringUTFChars(deviceName, nullptr);
  mm_handle_t detector{};
  auto ec =
      mmdeploy_rotated_detector_create_by_path(model_path, device_name, (int)device_id, &detector);
  env->ReleaseStringUTFChars(modelPath, model_path);
  env->ReleaseStringUTFChars(deviceName, device_name);
  if (ec) {
    MMDEPLOY_ERROR("failed to create rotated detector, code = {}", ec);
  }
  return (jlong)detector;
}

void Java_mmdeploy_RotatedDetector_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_INFO("Java_mmdeploy_RotatedDetector_destroy");  // maybe use info?
  mmdeploy_rotated_detector_destroy((mm_handle_t)handle);
}

jobjectArray Java_mmdeploy_RotatedDetector_apply(JNIEnv *env, jobject thiz, jlong handle,
                                                 jobjectArray images, jintArray counts) {
  return With(env, images, [&](const mm_mat_t imgs[], int size) {
    mm_rotated_detect_t *results{};
    int *result_count{};
    auto ec =
        mmdeploy_rotated_detector_apply((mm_handle_t)handle, imgs, size, &results, &result_count);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply rotated detector, code = {}", ec);
    }
    auto result_cls = env->FindClass("mmdeploy/RotatedDetector$Result");
    auto result_ctor = env->GetMethodID(result_cls, "<init>", "(IF[F)V");
    auto total = std::accumulate(result_count, result_count + size, 0);
    auto array = env->NewObjectArray(total, result_cls, nullptr);

    for (int i = 0; i < total; ++i) {
      auto rbbox = env->NewFloatArray(5);
      env->SetFloatArrayRegion(rbbox, 0, 5, (jfloat *)results[i].rbbox);
      auto res = env->NewObject(result_cls, result_ctor, (jint)results[i].label_id,
                                (jfloat)results[i].score, rbbox);
      env->SetObjectArrayElement(array, i, res);
    }
    auto counts_array = env->GetIntArrayElements(counts, nullptr);
    for (int i = 0; i < size; ++i) {
      counts_array[i] = result_count[i];
    }
    env->ReleaseIntArrayElements(counts, counts_array, 0);
    mmdeploy_rotated_detector_release_result(results, result_count);
    return array;
  });
}
