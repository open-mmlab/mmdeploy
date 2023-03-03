#include "mmdeploy_TextDetector.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/text_detector.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_TextDetector_create(JNIEnv *env, jobject, jstring modelPath, jstring deviceName,
                                        jint device_id) {
  auto model_path = env->GetStringUTFChars(modelPath, nullptr);
  auto device_name = env->GetStringUTFChars(deviceName, nullptr);
  mmdeploy_text_detector_t text_detector{};
  auto ec = mmdeploy_text_detector_create_by_path(model_path, device_name, (int)device_id,
                                                  &text_detector);
  env->ReleaseStringUTFChars(modelPath, model_path);
  env->ReleaseStringUTFChars(deviceName, device_name);
  if (ec) {
    MMDEPLOY_ERROR("failed to create text_detector, code = {}", ec);
    return -1;
  }
  return (jlong)text_detector;
}

void Java_mmdeploy_TextDetector_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_DEBUG("Java_mmdeploy_TextDetector_destroy");
  mmdeploy_text_detector_destroy((mmdeploy_text_detector_t)handle);
}

jobjectArray Java_mmdeploy_TextDetector_apply(JNIEnv *env, jobject thiz, jlong handle,
                                              jobjectArray images, jintArray counts) {
  return With(env, images, [&](const mmdeploy_mat_t imgs[], int size) -> jobjectArray {
    mmdeploy_text_detection_t *results{};
    int *result_count{};
    auto ec = mmdeploy_text_detector_apply((mmdeploy_text_detector_t)handle, imgs, size, &results,
                                           &result_count);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply detector, code = {}", ec);
      return NULL;
    }
    auto result_cls = env->FindClass("mmdeploy/TextDetector$Result");
    auto result_ctor = env->GetMethodID(result_cls, "<init>", "([Lmmdeploy/PointF;F)V");
    auto total = std::accumulate(result_count, result_count + size, 0);
    auto array = env->NewObjectArray(total, result_cls, nullptr);
    auto point_cls = env->FindClass("mmdeploy/PointF");
    auto point_ctor = env->GetMethodID(point_cls, "<init>", "(FF)V");

    for (int i = 0; i < total; ++i) {
      jobjectArray bbox = env->NewObjectArray(4, point_cls, nullptr);
      for (int j = 0; j < 4; ++j) {
        auto point = env->NewObject(point_cls, point_ctor, (jfloat)results[i].bbox[j].x,
                                    (jfloat)results[i].bbox[j].y);
        env->SetObjectArrayElement(bbox, j, point);
      }
      auto res = env->NewObject(result_cls, result_ctor, bbox, (jfloat)results[i].score);
      env->SetObjectArrayElement(array, i, res);
    }
    auto counts_array = env->GetIntArrayElements(counts, nullptr);
    for (int i = 0; i < size; ++i) {
      counts_array[i] = result_count[i];
    }
    env->ReleaseIntArrayElements(counts, counts_array, 0);
    mmdeploy_text_detector_release_result(results, result_count, size);
    return array;
  });
}
