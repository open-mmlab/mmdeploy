// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_detector.h"

#include <android/log.h>
#include <jni.h>
#include <stdio.h>
#include <unistd.h>

#include "common.h"

namespace MMDeploy {

extern "C" {
JNIEXPORT jobject JNICALL CreateByPath(JNIEnv* env, jobject thiz, jstring modelPath,
                                       jstring deviceName, jint deviceID, jobject handlePointer) {
  // handlePointer saves the value of mm_handle_t, because mm_handle_t is already a handle pointer.
  int status{};
  const char* model_path = env->GetStringUTFChars(modelPath, 0);
  const char* device_name = env->GetStringUTFChars(deviceName, 0);
  // handlePointer is a Java object which saves mm_handle_t pose detector address.
  jclass clazz = env->GetObjectClass(handlePointer);
  jmethodID initMethod = env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;J)V");
  jfieldID id_address = env->GetFieldID(clazz, "address", "J");
  mm_handle_t pose_detector{};
  int device_id = (int)deviceID;
  status =
      mmdeploy_pose_detector_create_by_path(model_path, device_name, device_id, &pose_detector);
  if (status != MM_SUCCESS) {
    __android_log_print(ANDROID_LOG_ERROR, "jni", "failed to create pose detector, code: %d\n",
                        (int)status);
    return env->NewObject(clazz, initMethod, env->NewStringUTF("mm_handle_t"), (long)0);
  }
  jobject result =
      env->NewObject(clazz, initMethod, env->NewStringUTF("mm_handle_t"), (long)pose_detector);
  return result;
}
JNIEXPORT jboolean JNICALL Apply(JNIEnv* env, jobject thiz, jobject handlePointer,
                                 jobject matsPointer, jint matCount, jobject resultsPointer) {
  int status{};
  jclass handle_clazz = env->GetObjectClass(handlePointer);
  jfieldID id_handle_address = env->GetFieldID(handle_clazz, "address", "J");
  long phandle = (long)env->GetLongField(handlePointer, id_handle_address);
  mm_handle_t pose_detector = (mm_handle_t)phandle;
  int mat_count = (int)matCount;
  jclass mats_clazz = env->GetObjectClass(matsPointer);
  jfieldID id_mats_address = env->GetFieldID(mats_clazz, "address", "J");
  // Here assume mats address is already save to cpp memory.
  jlong pmats = env->GetLongField(matsPointer, id_mats_address);
  jclass results_clazz = env->GetObjectClass(resultsPointer);
  jfieldID id_results_address = env->GetFieldID(results_clazz, "address", "J");
  jlong presults = env->GetLongField(resultsPointer, id_results_address);
  mm_pose_detect_t* result_apply = (mm_pose_detect_t*)presults;
  status =
      mmdeploy_pose_detector_apply(pose_detector, (const mm_mat_t*)pmats, mat_count, &result_apply);
  if (status != MM_SUCCESS) {
    __android_log_print(ANDROID_LOG_ERROR, "jni", "failed to create pose detector, code: %d\n",
                        (int)status);
    return JNI_FALSE;
  }
  env->SetLongField(resultsPointer, id_results_address, (jlong)result_apply);
  return JNI_TRUE;
}
JNIEXPORT void JNICALL ReleaseResult(JNIEnv* env, jobject thiz, jobject resultsPointer,
                                     jint count) {
  jclass results_clazz = env->GetObjectClass(resultsPointer);
  jfieldID id_results_address = env->GetFieldID(results_clazz, "address", "J");
  jlong presults = env->GetLongField(resultsPointer, id_results_address);
  mm_pose_detect_t* result = (mm_pose_detect_t*)presults;
  mmdeploy_pose_detector_release_result(result, (int)count);
}

JNIEXPORT void JNICALL Destroy(JNIEnv* env, jobject thiz, jobject handlePointer) {
  jclass handle_clazz = env->GetObjectClass(handlePointer);
  jfieldID id_handle_address = env->GetFieldID(handle_clazz, "address", "J");
  long phandle = (long)env->GetLongField(handlePointer, id_handle_address);
  mm_handle_t pose_detector = (mm_handle_t)phandle;
  mmdeploy_pose_detector_destroy(pose_detector);
}

static JNINativeMethod method[] = {
    {"CreateByPath",
     "(Ljava/lang/String;Ljava/lang/String;ILcn/org/openmmlab/mmdeploy/"
     "PointerWrapper;)Lcn/org/openmmlab/mmdeploy/PointerWrapper;",
     (bool*)CreateByPath},
    {"Apply",
     "(Lcn/org/openmmlab/mmdeploy/PointerWrapper;Lcn/org/openmmlab/mmdeploy/"
     "PointerWrapper;ILcn/org/openmmlab/mmdeploy/PointerWrapper;)Z",
     (bool*)Apply},
    {"ReleaseResult", "(Lcn/org/openmmlab/mmdeploy/PointerWrapper;I)V", (void*)ReleaseResult},
    {"Destroy", "(Lcn/org/openmmlab/mmdeploy/PointerWrapper;)V", (void*)Destroy}};
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  JNIEnv* env = NULL;
  jint result = -1;
  if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
    return result;
  }
  jclass jClassName = env->FindClass("cn/org/openmmlab/mmdeploy/PoseDetector/");
  jint ret = env->RegisterNatives(jClassName, method, sizeof(method) / sizeof(JNINativeMethod));
  if (ret != JNI_OK) {
    return -1;
  }
  return JNI_VERSION_1_6;
}
}
}  // namespace MMDeploy
