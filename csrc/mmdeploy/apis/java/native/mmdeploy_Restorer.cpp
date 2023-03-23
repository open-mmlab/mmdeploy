#include "mmdeploy_Restorer.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/restorer.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_Restorer_create(JNIEnv *env, jobject, jstring modelPath, jstring deviceName,
                                    jint device_id) {
  auto model_path = env->GetStringUTFChars(modelPath, nullptr);
  auto device_name = env->GetStringUTFChars(deviceName, nullptr);
  mmdeploy_restorer_t restorer{};
  auto ec = mmdeploy_restorer_create_by_path(model_path, device_name, (int)device_id, &restorer);
  env->ReleaseStringUTFChars(modelPath, model_path);
  env->ReleaseStringUTFChars(deviceName, device_name);
  if (ec) {
    MMDEPLOY_ERROR("failed to create restorer, code = {}", ec);
    return -1;
  }
  return (jlong)restorer;
}

void Java_mmdeploy_Restorer_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_DEBUG("Java_mmdeploy_Restorer_destroy");
  mmdeploy_restorer_destroy((mmdeploy_restorer_t)handle);
}

jobjectArray Java_mmdeploy_Restorer_apply(JNIEnv *env, jobject thiz, jlong handle,
                                          jobjectArray images) {
  return With(env, images, [&](const mmdeploy_mat_t imgs[], int size) -> jobjectArray {
    mmdeploy_mat_t *results{};
    auto ec = mmdeploy_restorer_apply((mmdeploy_restorer_t)handle, imgs, size, &results);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply restorer, code = {}", ec);
      return NULL;
    }
    const char *java_enum_format[] = {"BGR", "RGB", "GRAYSCALE", "NV12", "NV21", "BGRA"};
    const char *java_enum_type[] = {"FLOAT", "HALF", "INT8", "INT32"};
    auto result_cls = env->FindClass("mmdeploy/Restorer$Result");
    auto result_ctor = env->GetMethodID(result_cls, "<init>", "(Lmmdeploy/Mat;)V");
    auto array = env->NewObjectArray(size, result_cls, nullptr);
    auto mat_cls = env->FindClass("mmdeploy/Mat");
    auto mat_ctor =
        env->GetMethodID(mat_cls, "<init>", "(IIILmmdeploy/PixelFormat;Lmmdeploy/DataType;[B)V");
    auto format_cls = env->FindClass("mmdeploy/PixelFormat");
    auto type_cls = env->FindClass("mmdeploy/DataType");

    mmdeploy_mat_t *current_result = results;
    for (int i = 0; i < size; ++i) {
      auto test_format = current_result->format;
      auto jdata = env->NewByteArray(current_result->width * current_result->height *
                                     current_result->channel);
      env->SetByteArrayRegion(
          jdata, 0, current_result->width * current_result->height * current_result->channel,
          (const jbyte *)(current_result->data));
      jfieldID format_id = env->GetStaticFieldID(
          format_cls, java_enum_format[current_result->format], "Lmmdeploy/PixelFormat;");
      jobject format = env->GetStaticObjectField(format_cls, format_id);
      jfieldID type_id = env->GetStaticFieldID(type_cls, java_enum_type[current_result->type],
                                               "Lmmdeploy/DataType;");
      jobject type = env->GetStaticObjectField(type_cls, type_id);
      auto result_mat = env->NewObject(mat_cls, mat_ctor, (jint)(current_result->height),
                                       (jint)(current_result->width),
                                       (jint)(current_result->channel), format, type, jdata);
      auto res = env->NewObject(result_cls, result_ctor, result_mat);
      env->SetObjectArrayElement(array, i, res);
      current_result++;
    }
    mmdeploy_restorer_release_result(results, size);
    return array;
  });
}
