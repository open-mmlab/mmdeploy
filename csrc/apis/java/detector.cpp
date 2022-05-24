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
    JNIEXPORT jobject JNICALL mmdeployDetectorCreateByPath(JNIEnv* env, jobject thiz, jstring modelPath, jstring deviceName, jint deviceID, jobject handlePointer)
    {
      // handlePointer saves the value of mm_handle_t, because mm_handle_t is already a handle pointer.
      int status{};
      const char* model_path = env->GetStringUTFChars(modelPath, 0);
      const char* device_name = env->GetStringUTFChars(deviceName, 0);
      // handlePointer is a Java object which saves mm_handle_t detector address.
      jclass clazz = env->GetObjectClass(handlePointer);
      jmethodID initMethod = env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;J)V");
      jfieldID id_address = env->GetFieldID(clazz, "address", "J");
      mm_handle_t detector = new mm_handle_t;
      int device_id = (int)deviceID;
      status = mmdeploy_detector_create_by_path(model_path, device_name, device_id, &detector);
      if (status != MM_SUCCESS) {
        fprintf(stderr, "failed to create detector, code: %d\n", (int)status);
        return env->NewObject(clazz, initMethod, env->NewStringUTF("mm_handle_t"), (long)0);
      }
      jobject result=env->NewObject(clazz, initMethod, env->NewStringUTF("mm_handle_t"), (long)detector);
      return result;
    }
    JNIEXPORT jboolean JNICALL mmdeployDetectorApply(JNIEnv* env, jobject thiz, jobject handlePointer, jobject matsPointer, jint matCount, jobject resultsPointer, jobject resultCountPointer)
    {
      int status{};
      jclass handle_clazz = env->GetObjectClass(handlePointer);
      jfieldID id_handle_address = env->GetFieldID(handle_clazz, "address", "J");
      long phandle = (long)env->GetLongField(handlePointer, id_handle_address);
      mm_handle_t detector = (mm_handle_t) phandle;
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
      mm_detect_t * bboxes_apply = (mm_detect_t *) presults;
      int* count_apply = (int*)presult_count;
      status = mmdeploy_detector_apply(detector, (const mm_mat_t *) pmats, mat_count, &bboxes_apply, &count_apply);
      if (status != MM_SUCCESS) {
        fprintf(stderr, "failed to apply detector, code: %d\n", (int)status);
        return JNI_FALSE;
      }
      env->SetLongField(resultsPointer, id_results_address, (jlong)bboxes_apply);
      env->SetLongField(resultCountPointer, id_result_count_address, (jlong)count_apply);
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
      mm_detect_t * bbox = (mm_detect_t *)presults;
      int * bboxcount = (int*) presult_count;
      mmdeploy_detector_release_result(bbox, bboxcount, (int)count);
    }

    JNIEXPORT void JNICALL mmdeployDetectorDestroy(JNIEnv* env, jobject thiz, jobject handlePointer) {
      jclass handle_clazz = env->GetObjectClass(handlePointer);
      jfieldID id_handle_address = env->GetFieldID(handle_clazz, "address", "J");
      long phandle = (long)env->GetLongField(handlePointer, id_handle_address);
      mm_handle_t detector = (mm_handle_t)phandle;
      mmdeploy_detector_destroy(detector);
    }

    static JNINativeMethod method[]={
      {"mmdeployDetectorCreateByPath", "(Ljava/lang/String;Ljava/lang/String;ILcom/openmmlab/mmdeployxdetector/PointerWrapper;)Lcom/openmmlab/mmdeployxdetector/PointerWrapper;",(bool*)mmdeployDetectorCreateByPath},
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
