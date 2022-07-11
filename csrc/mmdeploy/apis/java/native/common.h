
#ifndef MMDEPLOY_CSRC_APIS_JAVA_NATIVE_COMMON_H_
#define MMDEPLOY_CSRC_APIS_JAVA_NATIVE_COMMON_H_

#include <jni.h>

#include <vector>

#include "mmdeploy/apis/c/common.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/utils/formatter.h"

template <typename F>
static auto With(JNIEnv *env, jobjectArray imgs, F f) noexcept {
  auto mat_clazz = env->FindClass("mmdeploy/Mat");
  auto shape_field = env->GetFieldID(mat_clazz, "shape", "[I");
  auto format_field = env->GetFieldID(mat_clazz, "format", "I");
  auto type_field = env->GetFieldID(mat_clazz, "type", "I");
  auto data_field = env->GetFieldID(mat_clazz, "data", "[B");
  auto num = env->GetArrayLength(imgs);
  std::vector<mm_mat_t> mats;
  std::vector<jbyteArray> datum;

  mats.reserve(num);
  datum.reserve(num);

  for (int i = 0; i < num; ++i) {
    auto obj = env->GetObjectArrayElement(imgs, i);
    auto shape_obj = env->GetObjectField(obj, shape_field);
    auto shape = env->GetIntArrayElements((jintArray)shape_obj, nullptr);
    auto format = env->GetIntField(obj, format_field);
    auto type = env->GetIntField(obj, type_field);
    auto &mat = mats.emplace_back();
    mat.height = shape[0];
    mat.width = shape[1];
    mat.channel = shape[2];
    env->ReleaseIntArrayElements((jintArray)shape_obj, shape, JNI_ABORT);
    mat.format = (mm_pixel_format_t)format;
    mat.type = (mm_data_type_t)type;
    auto data_obj = env->GetObjectField(obj, data_field);
    mat.data = (uint8_t *)env->GetByteArrayElements((jbyteArray)data_obj, nullptr);
    datum.push_back((jbyteArray)data_obj);
  }

  auto ret = f(mats.data(), mats.size());  // ! f must not throw

  for (int i = 0; i < num; ++i) {
    env->ReleaseByteArrayElements(datum[i], (jbyte *)mats[i].data, JNI_ABORT);
  }

  return ret;
}

#endif  // MMDEPLOY_CSRC_APIS_JAVA_NATIVE_COMMON_H_
