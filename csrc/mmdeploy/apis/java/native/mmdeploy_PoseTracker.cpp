#include "mmdeploy_PoseTracker.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/pose_tracker.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_PoseTracker_create(JNIEnv *env, jobject, jlong detModel, jlong poseModel,
                                       jlong context) {
  mmdeploy_pose_tracker_t pose_tracker{};
  auto ec = mmdeploy_pose_tracker_create((mmdeploy_model_t)detModel, (mmdeploy_model_t)poseModel,
                                         (mmdeploy_context_t)context, &pose_tracker);
  if (ec) {
    MMDEPLOY_ERROR("failed to create pose tracker, code = {}", ec);
    return -1;
  }
  return (jlong)pose_tracker;
}

void Java_mmdeploy_PoseTracker_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_DEBUG("Java_mmdeploy_PoseTracker_destroy");
  mmdeploy_pose_tracker_destroy((mmdeploy_pose_tracker_t)handle);
}

jobject param_cpp_to_java(JNIEnv *env, mmdeploy_pose_tracker_param_t *params) {
  auto param_cls = env->FindClass("mmdeploy/PoseTracker$Params");
  auto param_ctor = env->GetMethodID(param_cls, "<init>", "(IIFFFIFIFFF[FIFIIFF[F)V");

  jfloatArray keypointSigmas = env->NewFloatArray(params->keypoint_sigmas_size);
  env->SetFloatArrayRegion(keypointSigmas, 0, params->keypoint_sigmas_size,
                           (jfloat *)params->keypoint_sigmas);
  jfloatArray smoothParams = env->NewFloatArray(3);
  env->SetFloatArrayRegion(smoothParams, 0, 3, (jfloat *)params->smooth_params);

  auto param = env->NewObject(
      param_cls, param_ctor, (jint)params->det_interval, (jint)params->det_label,
      (jfloat)params->det_thr, (jfloat)params->det_min_bbox_size, (jfloat)params->det_nms_thr,
      (jint)params->pose_max_num_bboxes, (jfloat)params->pose_kpt_thr,
      (jint)params->pose_min_keypoints, (jfloat)params->pose_bbox_scale,
      (jfloat)params->pose_min_bbox_size, (jfloat)params->pose_nms_thr, keypointSigmas,
      (jint)params->keypoint_sigmas_size, (jfloat)params->track_iou_thr,
      (jint)params->track_max_missing, (jint)params->track_history_size,
      (jfloat)params->std_weight_position, (jfloat)params->std_weight_velocity, smoothParams);
  return param;
}

void param_java_to_cpp(JNIEnv *env, mmdeploy_pose_tracker_param_t *params, jobject customParam) {
  auto param_cls = env->FindClass("mmdeploy/PoseTracker$Params");
  auto param_ctor = env->GetMethodID(param_cls, "<init>", "(IIFFFIFIFFF[FIFIIFF[F)V");

  jfieldID fieldID_detInterval = env->GetFieldID(param_cls, "detInterval", "I");
  jint detInterval = env->GetIntField(customParam, fieldID_detInterval);
  params->det_interval = (int)detInterval;
  jfieldID fieldID_detLabel = env->GetFieldID(param_cls, "detLabel", "I");
  jint detLabel = env->GetIntField(customParam, fieldID_detLabel);
  params->det_label = (int)detLabel;
  jfieldID fieldID_detThr = env->GetFieldID(param_cls, "detThr", "F");
  jfloat detThr = env->GetFloatField(customParam, fieldID_detThr);
  params->det_thr = (float)detThr;
  jfieldID fieldID_detMinBboxSize = env->GetFieldID(param_cls, "detMinBboxSize", "F");
  jfloat detMinBboxSize = env->GetFloatField(customParam, fieldID_detMinBboxSize);
  params->det_min_bbox_size = (float)detMinBboxSize;
  jfieldID fieldID_detNmsThr = env->GetFieldID(param_cls, "detNmsThr", "F");
  jfloat detNmsThr = env->GetFloatField(customParam, fieldID_detNmsThr);
  params->det_nms_thr = (float)detNmsThr;
  jfieldID fieldID_poseMaxNumBboxes = env->GetFieldID(param_cls, "poseMaxNumBboxes", "I");
  jint poseMaxNumBboxes = env->GetIntField(customParam, fieldID_poseMaxNumBboxes);
  params->pose_max_num_bboxes = (int)poseMaxNumBboxes;
  jfieldID fieldID_poseKptThr = env->GetFieldID(param_cls, "poseKptThr", "F");
  jfloat poseKptThr = env->GetFloatField(customParam, fieldID_poseKptThr);
  params->pose_kpt_thr = (float)poseKptThr;
  jfieldID fieldID_poseMinKeypoints = env->GetFieldID(param_cls, "poseMinKeypoints", "I");
  jint poseMinKeypoints = env->GetIntField(customParam, fieldID_poseMinKeypoints);
  params->pose_min_keypoints = (int)poseMinKeypoints;
  jfieldID fieldID_poseBboxScale = env->GetFieldID(param_cls, "poseBboxScale", "F");
  jfloat poseBboxScale = env->GetFloatField(customParam, fieldID_poseBboxScale);
  params->pose_bbox_scale = (float)poseBboxScale;
  jfieldID fieldID_poseMinBboxSize = env->GetFieldID(param_cls, "poseMinBboxSize", "F");
  jfloat poseMinBboxSize = env->GetFloatField(customParam, fieldID_poseMinBboxSize);
  params->pose_min_bbox_size = (float)poseMinBboxSize;
  jfieldID fieldID_poseNmsThr = env->GetFieldID(param_cls, "poseNmsThr", "F");
  jfloat poseNmsThr = env->GetFloatField(customParam, fieldID_poseNmsThr);
  params->pose_nms_thr = (float)poseNmsThr;
  jfieldID fieldID_keypointSigmas = env->GetFieldID(param_cls, "keypointSigmas", "[F");
  auto keypointSigmasObj = env->GetObjectField(customParam, fieldID_keypointSigmas);
  float *keypointSigmas =
      (float *)env->GetFloatArrayElements((jfloatArray)keypointSigmasObj, nullptr);
  params->keypoint_sigmas = keypointSigmas;
  env->ReleaseFloatArrayElements((jfloatArray)keypointSigmasObj, keypointSigmas, JNI_ABORT);
  jfieldID fieldID_keypointSigmasSize = env->GetFieldID(param_cls, "keypointSigmasSize", "I");
  jint keypointSigmasSize = env->GetIntField(customParam, fieldID_keypointSigmasSize);
  params->keypoint_sigmas_size = keypointSigmasSize;
  jfieldID fieldID_trackIouThr = env->GetFieldID(param_cls, "trackIouThr", "F");
  jfloat trackIouThr = env->GetFloatField(customParam, fieldID_trackIouThr);
  params->track_iou_thr = trackIouThr;
  jfieldID fieldID_trackMaxMissing = env->GetFieldID(param_cls, "trackMaxMissing", "I");
  jint trackMaxMissing = env->GetIntField(customParam, fieldID_trackMaxMissing);
  params->track_max_missing = trackMaxMissing;
  jfieldID fieldID_trackHistorySize = env->GetFieldID(param_cls, "trackHistorySize", "I");
  jint trackHistorySize = env->GetIntField(customParam, fieldID_trackHistorySize);
  params->track_history_size = trackHistorySize;
  jfieldID fieldID_stdWeightPosition = env->GetFieldID(param_cls, "stdWeightPosition", "F");
  jfloat stdWeightPosition = env->GetFloatField(customParam, fieldID_stdWeightPosition);
  params->std_weight_position = stdWeightPosition;
  jfieldID fieldID_stdWeightVelocity = env->GetFieldID(param_cls, "stdWeightVelocity", "F");
  jfloat stdWeightVelocity = env->GetFloatField(customParam, fieldID_stdWeightVelocity);
  params->std_weight_velocity = stdWeightVelocity;
  jfieldID fieldID_smoothParams = env->GetFieldID(param_cls, "smoothParams", "[F");
  auto smoothParamsObj = env->GetObjectField(customParam, fieldID_smoothParams);
  float *smoothParams = (float *)env->GetFloatArrayElements((jfloatArray)smoothParamsObj, nullptr);
  params->smooth_params[0] = smoothParams[0];
  params->smooth_params[1] = smoothParams[1];
  params->smooth_params[2] = smoothParams[2];
  env->ReleaseFloatArrayElements((jfloatArray)smoothParamsObj, smoothParams, JNI_ABORT);
}

jobject Java_mmdeploy_PoseTracker_setDefaultParams(JNIEnv *env, jobject) {
  mmdeploy_pose_tracker_param_t params{};
  mmdeploy_pose_tracker_default_params(&params);
  return param_cpp_to_java(env, &params);
}

jlong Java_mmdeploy_PoseTracker_createState(JNIEnv *env, jobject, jlong pipeline,
                                            jobject paramsObject) {
  mmdeploy_pose_tracker_state_t state{};
  mmdeploy_pose_tracker_param_t params{};
  param_java_to_cpp(env, &params, paramsObject);
  auto ec = mmdeploy_pose_tracker_create_state((mmdeploy_pose_tracker_t)pipeline, &params, &state);
  if (ec) {
    MMDEPLOY_ERROR("failed to create pose tracker state, code = {}", ec);
    return -1;
  }
  return (jlong)state;
}

void Java_mmdeploy_PoseTracker_destroyState(JNIEnv *, jobject, jlong state) {
  MMDEPLOY_DEBUG("Java_mmdeploy_PoseTracker_destroy");
  mmdeploy_pose_tracker_destroy_state((mmdeploy_pose_tracker_state_t)state);
}

jobjectArray Java_mmdeploy_PoseTracker_apply(JNIEnv *env, jobject thiz, jlong handle,
                                             jlongArray states, jobjectArray frames,
                                             jintArray detects, jintArray counts) {
  return With(env, frames, [&](const mmdeploy_mat_t imgs[], int size) -> jobjectArray {
    mmdeploy_pose_tracker_target_t *results{};
    int *result_count{};
    auto states_array = env->GetLongArrayElements(states, nullptr);
    auto detects_array = env->GetIntArrayElements(detects, nullptr);
    auto ec = mmdeploy_pose_tracker_apply((mmdeploy_pose_tracker_t)handle,
                                          (mmdeploy_pose_tracker_state_t *)states_array, imgs,
                                          (int32_t *)detects_array, size, &results, &result_count);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply pose tracker, code = {}", ec);
      return NULL;
    }
    auto result_cls = env->FindClass("mmdeploy/PoseTracker$Result");
    auto result_ctor =
        env->GetMethodID(result_cls, "<init>", "([Lmmdeploy/PointF;[FLmmdeploy/Rect;I)V");
    auto total = std::accumulate(result_count, result_count + size, 0);
    auto array = env->NewObjectArray(total, result_cls, nullptr);
    auto pointf_cls = env->FindClass("mmdeploy/PointF");
    auto pointf_ctor = env->GetMethodID(pointf_cls, "<init>", "(FF)V");
    auto rect_cls = env->FindClass("mmdeploy/Rect");
    auto rect_ctor = env->GetMethodID(rect_cls, "<init>", "(FFFF)V");
    for (int i = 0; i < total; ++i) {
      auto keypoint_array = env->NewObjectArray(results[i].keypoint_count, pointf_cls, nullptr);
      for (int j = 0; j < results[i].keypoint_count; ++j) {
        auto keypointj = env->NewObject(pointf_cls, pointf_ctor, (jfloat)results[i].keypoints[j].x,
                                        (jfloat)results[i].keypoints[j].y);
        env->SetObjectArrayElement(keypoint_array, j, keypointj);
      }
      auto score_array = env->NewFloatArray(results[i].keypoint_count);
      env->SetFloatArrayRegion(score_array, 0, results[i].keypoint_count,
                               (jfloat *)results[i].scores);
      auto rect = env->NewObject(rect_cls, rect_ctor, (jfloat)results[i].bbox.left,
                                 (jfloat)results[i].bbox.top, (jfloat)results[i].bbox.right,
                                 (jfloat)results[i].bbox.bottom);
      auto target_id = results[i].target_id;
      auto res = env->NewObject(result_cls, result_ctor, keypoint_array, score_array, rect,
                                (int)target_id);
      env->SetObjectArrayElement(array, i, res);
    }
    auto counts_array = env->GetIntArrayElements(counts, nullptr);
    for (int i = 0; i < size; ++i) {
      counts_array[i] = result_count[i];
    }
    env->ReleaseIntArrayElements(counts, counts_array, 0);
    env->ReleaseLongArrayElements(states, states_array, 0);
    env->ReleaseIntArrayElements(detects, detects_array, 0);
    mmdeploy_pose_tracker_release_result(results, result_count, size);
    return array;
  });
}
