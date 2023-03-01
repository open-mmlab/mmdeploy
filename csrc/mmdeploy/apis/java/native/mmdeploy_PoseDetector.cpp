#include "mmdeploy_PoseDetector.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/pose_detector.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_PoseDetector_create(JNIEnv *env, jobject, jstring modelPath, jstring deviceName,
                                        jint device_id) {
  auto model_path = env->GetStringUTFChars(modelPath, nullptr);
  auto device_name = env->GetStringUTFChars(deviceName, nullptr);
  mmdeploy_pose_detector_t pose_estimator{};
  auto ec = mmdeploy_pose_detector_create_by_path(model_path, device_name, (int)device_id,
                                                  &pose_estimator);
  env->ReleaseStringUTFChars(modelPath, model_path);
  env->ReleaseStringUTFChars(deviceName, device_name);
  if (ec) {
    MMDEPLOY_ERROR("failed to create pose estimator, code = {}", ec);
    return -1;
  }
  return (jlong)pose_estimator;
}

void Java_mmdeploy_PoseDetector_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_DEBUG("Java_mmdeploy_PoseDetector_destroy");
  mmdeploy_pose_detector_destroy((mmdeploy_pose_detector_t)handle);
}

jobjectArray Java_mmdeploy_PoseDetector_apply(JNIEnv *env, jobject thiz, jlong handle,
                                              jobjectArray images) {
  return With(env, images, [&](const mmdeploy_mat_t imgs[], int size) -> jobjectArray {
    mmdeploy_pose_detection_t *results{};
    auto ec = mmdeploy_pose_detector_apply((mmdeploy_pose_detector_t)handle, imgs, size, &results);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply pose estimator, code = {}", ec);
      return NULL;
    }
    auto result_cls = env->FindClass("mmdeploy/PoseDetector$Result");
    auto result_ctor = env->GetMethodID(result_cls, "<init>", "([Lmmdeploy/PointF;[F)V");
    auto array = env->NewObjectArray(size, result_cls, nullptr);
    auto pointf_cls = env->FindClass("mmdeploy/PointF");
    auto pointf_ctor = env->GetMethodID(pointf_cls, "<init>", "(FF)V");

    for (int i = 0; i < size; ++i) {
      auto keypoint_array = env->NewObjectArray(results[i].length, pointf_cls, nullptr);
      for (int j = 0; j < results[i].length; ++j) {
        auto keypointj = env->NewObject(pointf_cls, pointf_ctor, (jfloat)results[i].point[j].x,
                                        (jfloat)results[i].point[j].y);
        env->SetObjectArrayElement(keypoint_array, j, keypointj);
      }
      auto score_array = env->NewFloatArray(results[i].length);
      env->SetFloatArrayRegion(score_array, 0, results[i].length, (jfloat *)results[i].score);
      auto res = env->NewObject(result_cls, result_ctor, keypoint_array, score_array);
      env->SetObjectArrayElement(array, i, res);
    }
    mmdeploy_pose_detector_release_result(results, size);
    return array;
  });
}
