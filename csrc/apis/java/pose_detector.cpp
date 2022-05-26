// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_detector.h"

#include "common.h"

#include <platform.h>
#include <benchmark.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace MMDeployJava {

  extern "C" {
    JNIEXPORT jobject JNICALL mmdeployPoseDetectorCreateByPath(JNIEnv* env, jobject thiz, jstring modelPath, jstring deviceName, jint deviceID, jobject handlePointer)
    {
      // handlePointer saves the value of mm_handle_t, because mm_handle_t is already a handle pointer.
      int status{};
      const char* model_path = env->GetStringUTFChars(modelPath, 0);
      const char* device_name = env->GetStringUTFChars(deviceName, 0);
      // handlePointer is a Java object which saves mm_handle_t pose detector address.
      jclass clazz = env->GetObjectClass(handlePointer);
      jmethodID initMethod = env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;J)V");
      jfieldID id_address = env->GetFieldID(clazz, "address", "J");
      mm_handle_t pose_detector = new mm_handle_t;
      int device_id = (int)deviceID;
      status = mmdeploy_pose_detector_create_by_path(model_path, device_name, device_id, &pose_detector);
      if (status != MM_SUCCESS) {
        fprintf(stderr, "failed to create pose_detector, code: %d\n", (int)status);
        return env->NewObject(clazz, initMethod, env->NewStringUTF("mm_handle_t"), (long)0);
      }
      jobject result=env->NewObject(clazz, initMethod, env->NewStringUTF("mm_handle_t"), (long)pose_detector);
      return result;
    }
    JNIEXPORT jboolean JNICALL mmdeployPoseDetectorApply(JNIEnv* env, jobject thiz, jobject handlePointer, jobject matsPointer, jint matCount, jobject resultsPointer)
    {
      int status{};
      jclass handle_clazz = env->GetObjectClass(handlePointer);
      jfieldID id_handle_address = env->GetFieldID(handle_clazz, "address", "J");
      long phandle = (long)env->GetLongField(handlePointer, id_handle_address);
      mm_handle_t pose_detector = (mm_handle_t) phandle;
      int mat_count = (int)matCount;
      jclass mats_clazz = env->GetObjectClass(matsPointer);
      jfieldID id_mats_address = env->GetFieldID(mats_clazz, "address", "J");
      // Here assume mats address is already save to cpp memory.
      jlong pmats = env->GetLongField(matsPointer, id_mats_address);
      jclass results_clazz = env->GetObjectClass(resultsPointer);
      jfieldID id_results_address = env->GetFieldID(results_clazz, "address", "J");
      jlong presults = env->GetLongField(resultsPointer, id_results_address);
      mm_pose_detect_t * result_apply = (mm_pose_detect_t *) presults;
      status = mmdeploy_pose_detector_apply(pose_detector, (const mm_mat_t *) pmats, mat_count, &result_apply);
      if (status != MM_SUCCESS) {
        fprintf(stderr, "failed to apply pose detector, code: %d\n", (int)status);
        return JNI_FALSE;
      }
      env->SetLongField(resultsPointer, id_results_address, (jlong)result_apply);
      return JNI_TRUE;
    }
    JNIEXPORT void JNICALL mmdeployPoseDetectorReleaseResult(JNIEnv* env, jobject thiz, jobject resultsPointer, jint count)
    {
      jclass results_clazz = env->GetObjectClass(resultsPointer);
      jfieldID id_results_address = env->GetFieldID(results_clazz, "address", "J");
      jlong presults = env->GetLongField(resultsPointer, id_results_address);
      mm_pose_detect_t * result = (mm_pose_detect_t *)presults;
      mmdeploy_pose_detector_release_result(result, (int)count);
    }

    JNIEXPORT void JNICALL mmdeployPoseDetectorDestroy(JNIEnv* env, jobject thiz, jobject handlePointer) {
      jclass handle_clazz = env->GetObjectClass(handlePointer);
      jfieldID id_handle_address = env->GetFieldID(handle_clazz, "address", "J");
      long phandle = (long)env->GetLongField(handlePointer, id_handle_address);
      mm_handle_t pose_detector = (mm_handle_t)phandle;
      mmdeploy_pose_detector_destroy(pose_detector);
    }

    static JNINativeMethod method[]={
      {"mmdeployPoseDetectorCreateByPath", "(Ljava/lang/String;Ljava/lang/String;ILcom/openmmlab/mmdeployxposedetector/PointerWrapper;)Lcom/openmmlab/mmdeployxposedetector/PointerWrapper;",(bool*)mmdeployPoseDetectorCreateByPath},
      {"mmdeployPoseDetectorApply", "(Lcom/openmmlab/mmdeployxposedetector/PointerWrapper;Lcom/openmmlab/mmdeployxposedetector/PointerWrapper;ILcom/openmmlab/mmdeployxposedetector/PointerWrapper;)Z",(bool*)mmdeployPoseDetectorApply},
      {"mmdeployPoseDetectorReleaseResult", "(Lcom/openmmlab/mmdeployxposedetector/PointerWrapper;I)V", (void*)mmdeployPoseDetectorReleaseResult},
      {"mmdeployPoseDetectorDestroy", "(Lcom/openmmlab/mmdeployxposedetector/PointerWrapper;)V", (void*)mmdeployPoseDetectorDestroy}
    };
    JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved){
      JNIEnv* env = NULL;
      jint result = -1;
      if(vm->GetEnv((void**)&env,JNI_VERSION_1_6)!= JNI_OK){
        return result;
      }
      jclass jClassName=env->FindClass("com/openmmlab/mmdeployxposedetector/MMDeployPoseDetector");
      jint ret = env->RegisterNatives(jClassName,method, sizeof(method)/sizeof(JNINativeMethod));
      if (ret != JNI_OK) {
          __android_log_print(ANDROID_LOG_DEBUG, "JNITag", "jni_register Error");
          return -1;
      }
      return JNI_VERSION_1_6;
    }
  }
}  // namespace mmdeployjava
