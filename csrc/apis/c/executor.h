//
// Created by zhangli on 4/26/22.
//

#ifndef MMDEPLOY_CSRC_APIS_C_EXECUTOR_H_
#define MMDEPLOY_CSRC_APIS_C_EXECUTOR_H_

#include "common.h"

#if __cplusplus
extern "C" {
#endif

typedef mmdeploy_value_t (*mmdeploy_invocable_t)(mmdeploy_value_t, void*);

struct mmdeploy_sender;
struct mmdeploy_scheduler;

typedef struct mmdeploy_sender* mmdeploy_sender_t;
typedef struct mmdeploy_scheduler* mmdeploy_scheduler_t;

typedef mmdeploy_sender_t (*mmdeploy_kleisli_t)(mmdeploy_value_t, void*);

typedef struct mmdeploy_exec_info* mmdeploy_exec_info_t;

struct mmdeploy_exec_info {
  mmdeploy_exec_info_t next;
  const char* task_name;
  mmdeploy_scheduler_t scheduler;
};

///////////////////////////////////////////////////////////////////////////////
// Scheduler
///////////////////////////////////////////////////////////////////////////////
MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_executor_inline();

MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_executor_system_pool();

MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_executor_create_thread_pool(int num_threads);

MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_executor_create_single_thread();

MMDEPLOY_API int mmdeploy_scheduler_destroy(mmdeploy_scheduler_t scheduler);

///////////////////////////////////////////////////////////////////////////////
// Sender factories
///////////////////////////////////////////////////////////////////////////////
MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler);

///////////////////////////////////////////////////////////////////////////////
// Sender adapters
///////////////////////////////////////////////////////////////////////////////
MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                                          mmdeploy_scheduler_t scheduler);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_on(mmdeploy_scheduler_t scheduler,
                                                    mmdeploy_sender_t input);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input,
                                                      mmdeploy_invocable_t fn, void* context);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_let_value(mmdeploy_sender_t input,
                                                           mmdeploy_kleisli_t kleisli,
                                                           void* context);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_when_all(mmdeploy_sender_t* inputs, int32_t n);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_ensure_started(mmdeploy_sender_t input);

///////////////////////////////////////////////////////////////////////////////
// Sender consumers
///////////////////////////////////////////////////////////////////////////////
MMDEPLOY_API int mmdeploy_executor_start_detached(mmdeploy_sender_t input);

MMDEPLOY_API mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input);

///////////////////////////////////////////////////////////////////////////////
// Utilities
///////////////////////////////////////////////////////////////////////////////
MMDEPLOY_API mmdeploy_sender_t mmdeploy_sender_copy(mmdeploy_sender_t input);

MMDEPLOY_API int mmdeploy_sender_destroy(mmdeploy_sender_t sender);

#if __cplusplus
}
#endif

#endif  // MMDEPLOY_CSRC_APIS_C_EXECUTOR_H_
