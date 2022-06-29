#include "mmdeploy_Detector.h"

#include <numeric>

#include "mmdeploy/apis/c/detector.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_Detector_create(JNIEnv *env, jobject, jstring modelPath, jstring deviceName,
                                    jint device_id) {
  auto model_path = env->GetStringUTFChars(modelPath, nullptr);
  auto device_name = env->GetStringUTFChars(deviceName, nullptr);
  mm_handle_t detector{};
  auto ec = mmdeploy_detector_create_by_path(model_path, device_name, (int)device_id, &detector);
  env->ReleaseStringUTFChars(modelPath, model_path);
  env->ReleaseStringUTFChars(deviceName, device_name);
  if (ec) {
    MMDEPLOY_ERROR("failed to create detector, code = {}", ec);
  }
  return (jlong)detector;
}

void Java_mmdeploy_Detector_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_INFO("Java_mmdeploy_Detector_destroy");  // maybe use info?
  mmdeploy_detector_destroy((mm_handle_t)handle);
}

jobjectArray Java_mmdeploy_Detector_apply(JNIEnv *env, jobject thiz, jlong handle,
                                          jobjectArray images, jintArray counts) {
  return With(env, images, [&](const mm_mat_t imgs[], int size) {
    mm_detect_t *results{};
    int *result_count{};
    auto ec = mmdeploy_detector_apply((mm_handle_t)handle, imgs, size, &results, &result_count);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply detector, code = {}", ec);
    }
    auto result_cls = env->FindClass("mmdeploy/Detector$Result");
    auto result_ctor =
        env->GetMethodID(result_cls, "<init>", "(IFLmmdeploy/Rect;Lmmdeploy/InstanceMask;)V");
    auto total = std::accumulate(result_count, result_count + size, 0);
    auto array = env->NewObjectArray(total, result_cls, nullptr);
    auto rect_cls = env->FindClass("mmdeploy/Rect");
    auto rect_ctor = env->GetMethodID(rect_cls, "<init>", "(FFFF)V");
    auto instance_mask_cls = env->FindClass("mmdeploy/InstanceMask");
    auto instance_mask_ctor = env->GetMethodID(instance_mask_cls, "<init>", "(II[C)V");

    for (int i = 0; i < total; ++i) {
      auto rect = env->NewObject(rect_cls, rect_ctor, (jfloat)results[i].bbox.left,
                                 (jfloat)results[i].bbox.top, (jfloat)results[i].bbox.right,
                                 (jfloat)results[i].bbox.bottom);
      int width, height;
      char *data;
      jcharArray jmask;
      if (results[i].mask == nullptr) {
        width = 0;
        height = 0;
        data = nullptr;
        jmask = env->NewCharArray(0);
      } else {
        width = results[i].mask->width;
        height = results[i].mask->height;
        data = results[i].mask->data;
        jmask = env->NewCharArray(width * height);
        env->SetCharArrayRegion(jmask, 0, width * height, (const jchar *)data);
      }

      auto instance_mask =
          env->NewObject(instance_mask_cls, instance_mask_ctor, (jint)height, (jint)width, jmask);
      auto res = env->NewObject(result_cls, result_ctor, (jint)results[i].label_id,
                                (jfloat)results[i].score, rect, instance_mask);
      env->SetObjectArrayElement(array, i, res);
    }
    auto counts_array = env->GetIntArrayElements(counts, nullptr);
    for (int i = 0; i < size; ++i) {
      counts_array[i] = result_count[i];
    }
    env->ReleaseIntArrayElements(counts, counts_array, 0);
    mmdeploy_detector_release_result(results, result_count, size);
    return array;
  });
}
