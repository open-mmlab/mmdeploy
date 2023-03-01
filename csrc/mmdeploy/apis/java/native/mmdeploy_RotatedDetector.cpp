#include "mmdeploy_RotatedDetector.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/rotated_detector.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_RotatedDetector_create(JNIEnv *env, jobject, jstring modelPath,
                                           jstring deviceName, jint device_id) {
  auto model_path = env->GetStringUTFChars(modelPath, nullptr);
  auto device_name = env->GetStringUTFChars(deviceName, nullptr);
  mmdeploy_rotated_detector_t rotated_detector{};
  auto ec = mmdeploy_rotated_detector_create_by_path(model_path, device_name, (int)device_id,
                                                     &rotated_detector);
  env->ReleaseStringUTFChars(modelPath, model_path);
  env->ReleaseStringUTFChars(deviceName, device_name);
  if (ec) {
    MMDEPLOY_ERROR("failed to create rotated detector, code = {}", ec);
    return -1;
  }
  return (jlong)rotated_detector;
}

void Java_mmdeploy_RotatedDetector_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_DEBUG("Java_mmdeploy_RotatedDetector_destroy");
  mmdeploy_rotated_detector_destroy((mmdeploy_rotated_detector_t)handle);
}

jobjectArray Java_mmdeploy_RotatedDetector_apply(JNIEnv *env, jobject thiz, jlong handle,
                                                 jobjectArray images, jintArray counts) {
  return With(env, images, [&](const mmdeploy_mat_t imgs[], int size) -> jobjectArray {
    mmdeploy_rotated_detection_t *results{};
    int *result_count{};
    auto ec = mmdeploy_rotated_detector_apply((mmdeploy_rotated_detector_t)handle, imgs, size,
                                              &results, &result_count);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply rotated detector, code = {}", ec);
      return NULL;
    }
    auto result_cls = env->FindClass("mmdeploy/RotatedDetector$Result");
    auto result_ctor = env->GetMethodID(result_cls, "<init>", "(IF[F)V");
    auto total = std::accumulate(result_count, result_count + size, 0);
    auto array = env->NewObjectArray(total, result_cls, nullptr);

    for (int i = 0; i < total; ++i) {
      jfloatArray rbbox = env->NewFloatArray(5);
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
