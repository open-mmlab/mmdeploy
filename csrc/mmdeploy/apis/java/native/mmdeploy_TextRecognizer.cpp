#include "mmdeploy_TextRecognizer.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/text_recognizer.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_TextRecognizer_create(JNIEnv *env, jobject, jstring modelPath,
                                          jstring deviceName, jint device_id) {
  auto model_path = env->GetStringUTFChars(modelPath, nullptr);
  auto device_name = env->GetStringUTFChars(deviceName, nullptr);
  mmdeploy_text_recognizer_t text_recognizer{};
  auto ec = mmdeploy_text_recognizer_create_by_path(model_path, device_name, (int)device_id,
                                                    &text_recognizer);
  env->ReleaseStringUTFChars(modelPath, model_path);
  env->ReleaseStringUTFChars(deviceName, device_name);
  if (ec) {
    MMDEPLOY_ERROR("failed to create text recognizer, code = {}", ec);
  }
  return (jlong)text_recognizer;
}

void Java_mmdeploy_TextRecognizer_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_DEBUG("Java_mmdeploy_TextRecognizer_destroy");  // maybe use info?
  mmdeploy_text_recognizer_destroy((mmdeploy_text_recognizer_t)handle);
}

jobjectArray Java_mmdeploy_TextRecognizer_apply(JNIEnv *env, jobject thiz, jlong handle,
                                                jobjectArray images) {
  return With(env, images, [&](const mmdeploy_mat_t imgs[], int size) {
    mmdeploy_text_recognition_t *results{};
    auto ec =
        mmdeploy_text_recognizer_apply((mmdeploy_text_recognizer_t)handle, imgs, size, &results);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply text recognizer, code = {}", ec);
    }
    auto result_cls = env->FindClass("mmdeploy/TextRecognizer$Result");
    auto result_ctor = env->GetMethodID(result_cls, "<init>", "([C[F)V");
    auto array = env->NewObjectArray(size, result_cls, nullptr);

    for (int i = 0; i < size; ++i) {
      auto text = env->NewCharArray(results[i].length);
      auto score = env->NewFloatArray(results[i].length);
      env->SetCharArrayRegion(text, 0, results[i].length, (jchar *)results[i].text);
      env->SetFloatArrayRegion(score, 0, results[i].length, (jfloat *)results[i].score);

      auto res = env->NewObject(result_cls, result_ctor, text, score);
      env->SetObjectArrayElement(array, i, res);
    }
    mmdeploy_text_recognizer_release_result(results, size);
    return array;
  });
}
jobjectArray Java_mmdeploy_TextRecognizer_applyBbox(JNIEnv *env, jobject thiz, jlong handle,
                                                    jobjectArray images, jobjectArray bboxes,
                                                    jintArray bbox_count) {
  return With(env, images, [&](const mmdeploy_mat_t imgs[], int size) {
    mmdeploy_text_recognition_t *recog_results{};
    auto *det_results = new mmdeploy_text_detection_t[env->GetArrayLength(bboxes)];
    int *det_result_count = new int[env->GetArrayLength(bbox_count)];
    auto bbox_cls = env->FindClass("mmdeploy/TextDetector$Result");
    auto pointf_cls = env->FindClass("mmdeploy/PointF");
    auto bbox_id = env->GetFieldID(bbox_cls, "bbox", "[Lmmdeploy/PointF;");
    auto score_id = env->GetFieldID(bbox_cls, "score", "F");
    auto x_id = env->GetFieldID(pointf_cls, "x", "F");
    auto y_id = env->GetFieldID(pointf_cls, "y", "F");
    env->GetIntArrayRegion(bbox_count, 0, env->GetArrayLength(bbox_count), det_result_count);
    int total_bboxes = env->GetArrayLength(bboxes);
    for (int i = 0; i < total_bboxes; ++i) {
      auto bboxi = env->GetObjectArrayElement(bboxes, i);
      auto point_array = (jobjectArray)env->GetObjectField(bboxi, bbox_id);
      for (int j = 0; j < 4; ++j) {
        auto pointj = env->GetObjectArrayElement(point_array, j);
        det_results[i].bbox[j].x = (float)env->GetFloatField(pointj, x_id);
        det_results[i].bbox[j].y = (float)env->GetFloatField(pointj, y_id);
        det_results[i].score = (float)env->GetFloatField(bboxi, score_id);
      }
    }
    auto ec = mmdeploy_text_recognizer_apply_bbox((mmdeploy_text_recognizer_t)handle, imgs, size,
                                                  (const mmdeploy_text_detection_t *)det_results,
                                                  det_result_count, &recog_results);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply bbox for text recognizer, code = {}", ec);
    }
    auto result_cls = env->FindClass("mmdeploy/TextRecognizer$Result");
    auto result_ctor = env->GetMethodID(result_cls, "<init>", "([B[F)V");
    auto array = env->NewObjectArray(total_bboxes, result_cls, nullptr);

    for (int i = 0; i < total_bboxes; ++i) {
      auto text = env->NewByteArray(recog_results[i].length);
      auto score = env->NewFloatArray(recog_results[i].length);
      env->SetByteArrayRegion(text, 0, recog_results[i].length, (jbyte *)recog_results[i].text);
      env->SetFloatArrayRegion(score, 0, recog_results[i].length, (jfloat *)recog_results[i].score);

      auto res = env->NewObject(result_cls, result_ctor, text, score);
      env->SetObjectArrayElement(array, i, res);
    }
    mmdeploy_text_recognizer_release_result(recog_results, size);
    mmdeploy_text_detector_release_result(det_results, det_result_count, 1);
    return array;
  });
}
