// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_APIS_C_EXECUTOR_H_
#define MMDEPLOY_CSRC_APIS_C_EXECUTOR_H_

#include "common.h"

#if __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Experimental asynchronous APIs */

typedef mmdeploy_value_t (*mmdeploy_then_fn_t)(mmdeploy_value_t, void*);

typedef mmdeploy_value_t (*mmdeploy_then_fn_v2_t)(mmdeploy_value_t*, void*);

typedef int (*mmdeploy_then_fn_v3_t)(mmdeploy_value_t* input, mmdeploy_value_t* output, void*);

struct mmdeploy_sender;
struct mmdeploy_scheduler;

typedef struct mmdeploy_sender* mmdeploy_sender_t;
typedef struct mmdeploy_scheduler* mmdeploy_scheduler_t;

typedef mmdeploy_sender_t (*mmdeploy_let_value_fn_t)(mmdeploy_value_t, void*);

///////////////////////////////////////////////////////////////////////////////
// Scheduler
///////////////////////////////////////////////////////////////////////////////
MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_executor_inline();

MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_executor_system_pool();

/**
 * Create a thread pool with the given number of worker threads
 * @param[in] num_threads
 * @return the handle to the created thread pool
 */
MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_executor_create_thread_pool(int num_threads);

MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_executor_create_thread();

MMDEPLOY_API mmdeploy_scheduler_t mmdeploy_executor_dynamic_batch(mmdeploy_scheduler_t scheduler,
                                                                  int max_batch_size, int timeout);

MMDEPLOY_API int mmdeploy_scheduler_destroy(mmdeploy_scheduler_t scheduler);

///////////////////////////////////////////////////////////////////////////////
// Utilities
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Create a copy of a copyable sender. Only senders created by \ref mmdeploy_executor_split
 * is copyable for now.
 * @param[in] input copyable sender,
 * @return the sender created, or nullptr if the sender is not copyable
 */
MMDEPLOY_API mmdeploy_sender_t mmdeploy_sender_copy(mmdeploy_sender_t input);

/**
 * @brief Destroy a sender, notice that all sender adapters will consume input senders, only unused
 * senders should be destroyed using this function.
 * @param[in] input
 */
MMDEPLOY_API int mmdeploy_sender_destroy(mmdeploy_sender_t sender);

///////////////////////////////////////////////////////////////////////////////
// Sender factories
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Create a sender that sends the provided value
 * @param[in] value
 * @return created sender
 */
MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value);

/**
 * @brief
 * @param[in] scheduler
 * @return the sender created
 */
MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_transfer_just(mmdeploy_scheduler_t scheduler,
                                                               mmdeploy_value_t value);

///////////////////////////////////////////////////////////////////////////////
// Sender adapters
///////////////////////////////////////////////////////////////////////////////

/**
 * Transfer the execution to the execution agent of the provided scheduler
 * @param[in] input
 * @param[in] scheduler
 * @return the sender created
 */
MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                                          mmdeploy_scheduler_t scheduler);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_on(mmdeploy_scheduler_t scheduler,
                                                    mmdeploy_sender_t input);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input,
                                                      mmdeploy_then_fn_t fn, void* context);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_let_value(mmdeploy_sender_t input,
                                                           mmdeploy_let_value_fn_t fn,
                                                           void* context);

/**
 * Convert the input sender into a sender that is copyable via \ref mmdeploy_sender_copy. Notice
 * that this function doesn't make the sender multi-shot, it just return a sender that is copyable.
 * @param[in] input
 * @return the sender that is copyable
 */
MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_when_all(mmdeploy_sender_t inputs[], int32_t n);

MMDEPLOY_API mmdeploy_sender_t mmdeploy_executor_ensure_started(mmdeploy_sender_t input);

///////////////////////////////////////////////////////////////////////////////
// Sender consumers
///////////////////////////////////////////////////////////////////////////////
MMDEPLOY_API int mmdeploy_executor_start_detached(mmdeploy_sender_t input);

MMDEPLOY_API mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input);

MMDEPLOY_API int mmdeploy_executor_sync_wait_v2(mmdeploy_sender_t input, mmdeploy_value_t* output);

MMDEPLOY_API void mmdeploy_executor_execute(mmdeploy_scheduler_t scheduler, void (*fn)(void*),
                                            void* context);

#if __cplusplus
}
#endif

#endif  // MMDEPLOY_CSRC_APIS_C_EXECUTOR_H_
