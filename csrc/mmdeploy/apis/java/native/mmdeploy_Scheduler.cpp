#include "mmdeploy_Scheduler.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/common.h"
#include "mmdeploy/apis/c/mmdeploy/executor.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_Scheduler_createThreadPool(JNIEnv *env, jobject, jint numThreads) {
  mmdeploy_scheduler_t scheduler = mmdeploy_executor_create_thread_pool((int)numThreads);
  return (jlong)scheduler;
}

jlong Java_mmdeploy_Scheduler_createThread(JNIEnv *env, jobject) {
  mmdeploy_scheduler_t scheduler = mmdeploy_executor_create_thread();
  return (jlong)scheduler;
}

void Java_mmdeploy_Scheduler_destroy(JNIEnv *, jobject, jlong scheduler_) {
  MMDEPLOY_DEBUG("Java_mmdeploy_Scheduler_destroy");
  mmdeploy_scheduler_destroy((mmdeploy_scheduler_t)scheduler_);
}
