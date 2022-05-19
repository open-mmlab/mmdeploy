// Copyright (c) OpenMMLab. All rights reserved.

#include "detector.h"

#include "common.h"

#include <platform.h>
#include <benchmark.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace MMDeployJava {

  extern "C" {
    JNIEXPORT jboolean JNICALL mmdeployDetectorCreateByPath(JNIEnv* env, jobject thiz, jstring modelPath, jstring deviceName, jint deviceID, jobject handlePointer)
    {
      int status{};
      const char* model_path = env->GetStringUTFChars(modelPath, 0);
      const char* device_name = env->GetStringUTFChars(deviceName, 0);
      // handlePointer is a Java class which saves mm_handle_t detector address.
      jclass clazz = env->GetObjectClass(handlePointer);
      jfieldID id_address = env->GetFieldID(clazz, "address", "J");
      jlong pdetector = env->GetLongField(handlePointer, id_address);
      // if detector is created from cpp data preparation, this if should be removed.
      if ((long) pdetector == 0) {
        mm_handle_t detector = new mm_handle_t; //use heap.
        pdetector = (jlong) &detector;
      }
      int device_id = (int)deviceID;
      status = mmdeploy_detector_create_by_path(model_path, device_name, device_id, (mm_handle_t*) pdetector);
      if (status != MM_SUCCESS) {
        fprintf(stderr, "failed to create detector, code: %d\n", (int)status);
        return JNI_FALSE;
      }
      env->SetLongField(handlePointer, id_address, pdetector);
      return JNI_TRUE;
    }
    JNIEXPORT jboolean JNICALL mmdeployDetectorApply(JNIEnv* env, jobject thiz, jobject handle, jobject matsPointer, jint matCount, jobject resultsPointer, jobject resultCountPointer)
    {
      int status{};
      mm_handle_t detector = (mm_handle_t)handle;
      int mat_count = (int)matCount;
      jclass mats_clazz = env->GetObjectClass(matsPointer);
      jfieldID id_mats_address = env->GetFieldID(mats_clazz, "address", "J");
      // Here assume mats address is already save to cpp memory.
      jlong pmats = env->GetLongField(matsPointer, id_mats_address);
      jclass results_clazz = env->GetObjectClass(resultsPointer);
      jfieldID id_results_address = env->GetFieldID(results_clazz, "address", "J");
      jlong presults = env->GetLongField(resultsPointer, id_results_address);
      jclass result_count_clazz = env->GetObjectClass(resultCountPointer);
      jfieldID id_result_count_address = env->GetFieldID(result_count_clazz, "address", "J");
      jlong presult_count = env->GetLongField(resultCountPointer, id_result_count_address);
      status = mmdeploy_detector_apply(detector, (const mm_mat_t *) pmats, mat_count, (mm_detect_t **)presults, (int **)presult_count);
      if (status != MM_SUCCESS) {
        fprintf(stderr, "failed to apply detector, code: %d\n", (int)status);
        return JNI_FALSE;
      }
      return JNI_TRUE;
    }
    JNIEXPORT void JNICALL mmdeployDetectorReleaseResult(JNIEnv* env, jobject thiz, jobject resultsPointer, jobject resultCountPointer, jint count)
    {
      jclass results_clazz = env->GetObjectClass(resultsPointer);
      jfieldID id_results_address = env->GetFieldID(results_clazz, "address", "J");
      jlong presults = env->GetLongField(resultsPointer, id_results_address);
      jclass results_count_clazz = env->GetObjectClass(resultCountPointer);
      jfieldID id_result_count_address = env->GetFieldID(results_count_clazz, "address", "J");
      jlong presult_count = env->GetLongField(resultCountPointer, id_result_count_address);
      mmdeploy_detector_release_result((mm_detect_t *)presults, (const int *)presult_count, (int)count);
    }

    JNIEXPORT void JNICALL mmdeployDetectorDestroy(JNIEnv* env, jobject thiz, jobject handle) {
      mm_handle_t detector = (mm_handle_t) handle;
      mmdeploy_detector_destroy(handle);
    }

    static JNINativeMethod method[]={
      {"mmdeployDetectorCreateByPath", "(Ljava/lang/String;Ljava/lang/String;ILcom/openmmlab/mmdeployxdetector/PointerWrapper;)Z",(bool*)mmdeployDetectorCreateByPath},
      {"mmdeployDetectorApply", "(Lcom/openmmlab/mmdeployxdetector/PointerWrapper;Lcom/openmmlab/mmdeployxdetector/PointerWrapper;ILcom/openmmlab/mmdeployxdetector/PointerWrapper;Lcom/openmmlab/mmdeployxdetector/PointerWrapper;)Z",(bool*)mmdeployDetectorApply},
      {"mmdeployDetectorReleaseResult", "(Lcom/openmmlab/mmdeployxdetector/PointerWrapper;Lcom/openmmlab/mmdeployxdetector/PointerWrapper;I)V", (void*)mmdeployDetectorReleaseResult},
      {"mmdeployDetectorDestroy", "(Lcom/openmmlab/mmdeployxdetector/PointerWrapper;)V", (void*)mmdeployDetectorDestroy}
    };
    JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved){
      JNIEnv* env = NULL;
      jint result = -1;
      if(vm->GetEnv((void**)&env,JNI_VERSION_1_6)!= JNI_OK){
        return result;
      }
      jclass jClassName=env->FindClass("com/openmmlab/mmdeployxdetector/MMDeployDetector");
      jint ret = env->RegisterNatives(jClassName,method, sizeof(method)/sizeof(JNINativeMethod));
      if (ret != JNI_OK) {
          __android_log_print(ANDROID_LOG_DEBUG, "JNITag", "jni_register Error");
          return -1;
      }
      return JNI_VERSION_1_6;
    }
  }
}  // namespace mmdeployjava
