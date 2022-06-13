// Copyright (c) OpenMMLab. All rights reserved.

#include "pipeline.h"

#include "mmdeploy/apis/c/common_internal.h"
#include "mmdeploy/apis/c/executor_internal.h"
#include "mmdeploy/apis/c/handle.h"

int mmdeploy_pipeline_create(mmdeploy_value_t config, const char* device_name, int device_id,
                             mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  try {
    auto _config = *Cast(config);
    if (exec_info) {
      auto& info = _config["context"]["executor"] = Value::kObject;
      for (auto p = exec_info; p; p = p->next) {
        info[p->task_name] = *Cast(p->scheduler);
        if (p->next == exec_info) {
          MMDEPLOY_ERROR("circle detected in exec_info list.");
          return MM_E_INVALID_ARG;
        }
      }
    }
    auto _handle = std::make_unique<AsyncHandle>(device_name, device_id, std::move(_config));
    *handle = _handle.release();
    return MM_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

int mmdeploy_pipeline_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                  mmdeploy_sender_t* output) {
  if (!handle || !input || !output) {
    return MM_E_INVALID_ARG;
  }
  try {
    auto h = static_cast<AsyncHandle*>(handle);
    *output = Take(h->Process(Take(input)));
    return MM_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

void mmdeploy_pipeline_destroy(mm_handle_t handle) {
  if (handle != nullptr) {
    delete static_cast<AsyncHandle*>(handle);
  }
}

int mmdeploy_pipeline_apply(mm_handle_t handle, mmdeploy_value_t input, mmdeploy_value_t* output) {
  auto input_sender = mmdeploy_executor_just(input);
  if (!input_sender) {
    return MM_E_FAIL;
  }
  mmdeploy_sender_t output_sender{};
  if (auto ec = mmdeploy_pipeline_apply_async(handle, input_sender, &output_sender)) {
    return ec;
  }
  auto _output = mmdeploy_executor_sync_wait(output_sender);
  if (!_output) {
    return MM_E_FAIL;
  }
  *output = _output;
  return MM_SUCCESS;
}
