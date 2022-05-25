// Copyright (c) OpenMMLab. All rights reserved.

#include "classifier.h"

#include "common.h"

#include <platform.h>
#include <benchmark.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace MMDeployJava {

  extern "C" {
    JNIEXPORT jobject JNICALL mmdeployClassifierCreateByPath(JNIEnv* env, jobject thiz, jstring modelPath, jstring deviceName, jint deviceID, jobject handlePointer)
    {
      // handlePointer saves the value of mm_handle_t, because mm_handle_t is already a handle pointer.
      int status{};
      const char* model_path = env->GetStringUTFChars(modelPath, 0);
      const char* device_name = env->GetStringUTFChars(deviceName, 0);
      // handlePointer is a Java object which saves mm_handle_t classifier address.
      jclass clazz = env->GetObjectClass(handlePointer);
      jmethodID initMethod = env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;J)V");
      jfieldID id_address = env->GetFieldID(clazz, "address", "J");
      mm_handle_t classifier = new mm_handle_t;
      int device_id = (int)deviceID;
      status = mmdeploy_classifier_create_by_path(model_path, device_name, device_id, &classifier);
      if (status != MM_SUCCESS) {
        fprintf(stderr, "failed to create classifier, code: %d\n", (int)status);
        return env->NewObject(clazz, initMethod, env->NewStringUTF("mm_handle_t"), (long)0);
      }
      jobject result=env->NewObject(clazz, initMethod, env->NewStringUTF("mm_handle_t"), (long)classifier);
      return result;
    }
    JNIEXPORT jboolean JNICALL mmdeployClassifierApply(JNIEnv* env, jobject thiz, jobject handlePointer, jobject matsPointer, jint matCount, jobject resultsPointer, jobject resultCountPointer)
    {
      int status{};
      jclass handle_clazz = env->GetObjectClass(handlePointer);
      jfieldID id_handle_address = env->GetFieldID(handle_clazz, "address", "J");
      long phandle = (long)env->GetLongField(handlePointer, id_handle_address);
      mm_handle_t classifier = (mm_handle_t) phandle;
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
      mm_class_t * res_apply = (mm_class_t *) presults;
      int* count_apply = (int*)presult_count;
      status = mmdeploy_classifier_apply(classifier, (const mm_mat_t *) pmats, mat_count, &res_apply, &count_apply);
      if (status != MM_SUCCESS) {
        fprintf(stderr, "failed to apply classifier, code: %d\n", (int)status);
        return JNI_FALSE;
      }
      env->SetLongField(resultsPointer, id_results_address, (jlong)res_apply);
      env->SetLongField(resultCountPointer, id_result_count_address, (jlong)count_apply);
      return JNI_TRUE;
    }
    JNIEXPORT void JNICALL mmdeployClassifierReleaseResult(JNIEnv* env, jobject thiz, jobject resultsPointer, jobject resultCountPointer, jint count)
    {
      jclass results_clazz = env->GetObjectClass(resultsPointer);
      jfieldID id_results_address = env->GetFieldID(results_clazz, "address", "J");
      jlong presults = env->GetLongField(resultsPointer, id_results_address);
      jclass results_count_clazz = env->GetObjectClass(resultCountPointer);
      jfieldID id_result_count_address = env->GetFieldID(results_count_clazz, "address", "J");
      jlong presult_count = env->GetLongField(resultCountPointer, id_result_count_address);
      mm_class_t * res = (mm_class_t *)presults;
      int * rescount = (int*) presult_count;
      mmdeploy_classifier_release_result(res, rescount, (int)count);
    }

    JNIEXPORT void JNICALL mmdeployClassifierDestroy(JNIEnv* env, jobject thiz, jobject handlePointer) {
      jclass handle_clazz = env->GetObjectClass(handlePointer);
      jfieldID id_handle_address = env->GetFieldID(handle_clazz, "address", "J");
      long phandle = (long)env->GetLongField(handlePointer, id_handle_address);
      mm_handle_t classifier = (mm_handle_t)phandle;
      mmdeploy_classifier_destroy(classifier);
    }

    static JNINativeMethod method[]={
      {"mmdeployClassifierCreateByPath", "(Ljava/lang/String;Ljava/lang/String;ILcom/openmmlab/mmdeployxclassifier/PointerWrapper;)Lcom/openmmlab/mmdeployxclassifier/PointerWrapper;",(bool*)mmdeployClassifierCreateByPath},
      {"mmdeployClassifierApply", "(Lcom/openmmlab/mmdeployxclassifier/PointerWrapper;Lcom/openmmlab/mmdeployxclassifier/PointerWrapper;ILcom/openmmlab/mmdeployxclassifier/PointerWrapper;Lcom/openmmlab/mmdeployxclassifier/PointerWrapper;)Z",(bool*)mmdeployClassifierApply},
      {"mmdeployClassifierReleaseResult", "(Lcom/openmmlab/mmdeployxclassifier/PointerWrapper;Lcom/openmmlab/mmdeployxclassifier/PointerWrapper;I)V", (void*)mmdeployClassifierReleaseResult},
      {"mmdeployClassifierDestroy", "(Lcom/openmmlab/mmdeployxclassifier/PointerWrapper;)V", (void*)mmdeployClassifierDestroy}
    };
    JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved){
      JNIEnv* env = NULL;
      jint result = -1;
      if(vm->GetEnv((void**)&env,JNI_VERSION_1_6)!= JNI_OK){
        return result;
      }
      jclass jClassName=env->FindClass("com/openmmlab/mmdeployxclassifier/MMDeployClassifier");
      jint ret = env->RegisterNatives(jClassName,method, sizeof(method)/sizeof(JNINativeMethod));
      if (ret != JNI_OK) {
          __android_log_print(ANDROID_LOG_DEBUG, "JNITag", "jni_register Error");
          return -1;
      }
      return JNI_VERSION_1_6;
    }
  }
}  // namespace mmdeployjava
